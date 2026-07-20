# ExecuTorch 对 Qwen3-4B 的量化方法（QNN/HTP 后端）

> 面向 Qwen3-4B 在 Qualcomm QNN / HTP 上的部署。目标模型（target）是 16a4w 量化，DFlash 草稿模型（draft）是 fp16 不量化。本文只讲 **target 的量化**。

## 0. 一句话概览

Qwen3-4B 用的是 **静态 PTQ（post-training quantization）+ 16a4w** 方案：**激活 16-bit（uint16 范围）逐张量（per-tensor）非对称，权重 4-bit（int4，[-7,7]）逐块（per-block，16 个输入通道一组）对称**，少数敏感层（lm_head、MLP down_proj）提到 16a8w 逐通道，KV cache 压到 8-bit。**词嵌入表也量化**：它走默认档 `use_16a4w PER_TENSOR`，但因为 `aten.embedding` 不是 conv/linear，那张表被当**输入激活**量化成 **uint16（16-bit）per-tensor**（`4w` 的 int4 只作用于 conv/linear 权重，作用不到它；见 §5.2）。全流程是标准的 `prepare_pt2e → 校准 → convert_pt2e`，产物再交给 QNN 后端 lower 成 HTP 二进制。

| 部件 | 位宽 | 粒度 | 对称性 | 容器 dtype / 范围 |
|---|---|---|---|---|
| 激活（默认） | 16-bit | per-tensor | 非对称（有 zero_point） | int32 容器，取值 uint16 `[0, 65535]` |
| 权重（Transformer 各投影，即 conv2d） | 4-bit | per-block `(1,16,1,1)` | 对称 | int8 容器，`[-7, 7]`，LPBQ |
| 权重（lm_head `output.conv` + MLP down_proj `w2_conv`） | 8-bit | per-channel | 对称 | int8，`[-127,127]` 级 |
| KV cache（K/V 分支） | 8-bit | per-tensor | 对称 | uint8 `[0, 255]` |
| Q·Kᵀ·V 的 matmul | 16a8w | — | 激活非对称 / 权重对称 | act uint16 / weight int8 |
| 词嵌入表 tok_embeddings | 16-bit | per-tensor | 非对称 | uint16 `[0,65535]`,走默认档(表被当激活量化,非 int4 权重;见 §5.2) |
| logits 输出 | 16-bit | per-tensor | 非对称 | uint16 出图，再 dequant 成 fp16 |

模型结构（`models/qwen3/config/4b_config.json`）：36 层，`dim=2560`，`hidden_dim=9728`，32 头 / 8 KV 头，`head_dim=128`，`vocab=151936`，带 QK-RMSNorm。

---

## 1. 关键前提：所有 Linear 都是 1×1 Conv2d

这个 codebase 把 Transformer 里所有的 `nn.Linear` 都重写成了 `nn.Conv2d(in, out, kernel_size=1)`：

- 注意力投影 `wq_conv / wk_conv / wv_conv / wo_conv` — `model/static_llama.py:348-366`
- MLP `gate_up_proj_conv / down_proj_conv`（或 `w1/w2/w3_conv`）— `model/feed_forward.py:59-112`、`model/static_llama.py:544-546`
- LM head `output_conv` — `model/static_llama.py:715`

所以量化 recipe 里 **一句 `torch.ops.aten.conv2d.default` 就覆盖了整个 Transformer 的所有权重矩阵**。这也是为什么 recipe 看起来只处理 conv2d —— 因为对这个模型来说 conv2d 就是全部的 GEMM。

---

## 2. 位宽枚举与配置

`backends/qualcomm/quantizer/quantizer.py:80-178` 定义了 `QuantDtype` 及其到具体 qconfig 的映射：

| 枚举 | 激活 | 权重 |
|---|---|---|
| `use_16a16w` | uint16 | int16 |
| `use_16a8w` | uint16 | int8 |
| **`use_16a4w`** | **uint16** | **int4** |
| **`use_16a4w_block`** | **uint16** | **int4（逐块）** |
| `use_8a8w` | uint8 | int8 |
| `use_8a4w` | uint8 | int4 |

具体范围（`backends/qualcomm/quantizer/qconfig.py`）：

- **16a4w 激活**（`qconfig.py:223-264`）：`dtype=torch.int32` 容器，`quant_min=0`、`quant_max=65535`（即 uint16 的范围）。用 int32 容器是因为 “torch 不支持 uint16 量化”，**不是 int16**。`qscheme=per_tensor_affine`（非对称，除非显式 `act_symmetric=True`）。
- **4-bit 权重**：`dtype=torch.int8` 容器，`quant_min=-7`、`quant_max=7`（int4 的 `[-7,7]`，**对称**，注意不是 `[-8,7]`），`ch_axis=0`。
- **bias**：int32，全 int32 范围，对称。

---

## 3. Qwen3-4B 专用 recipe

`examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py:646-693`，`Qwen3_4BQuantRecipe`：

```python
default_quant_dtype = QuantDtype.use_16a4w          # 默认 per-tensor：act uint16 + weight int4

self.recipe = (
    QuantRecipe(use_16a4w, PER_TENSOR, MinMaxObserver)   # 兜底策略
    .add_node_target(                                    # 所有 conv2d（= 所有 Linear）
        {aten.conv2d.default},
        use_16a4w_block, PER_BLOCK,
        extra_kwargs={"block_size": (1, 16, 1, 1)},
    )
    .add_regex(                                           # 敏感层升到 16a8w 逐通道
        {r"output\.conv", r"layers\..*\.feed_forward\.w2_conv"},
        use_16a8w, PER_CHANNEL,
    )
)
self.recipe.custom_quant_annotations.append(annotate_kv_8bit)   # KV → 8-bit
```

策略优先级（`quant_recipe.py:274-313`）：正则匹配 > node-target 匹配 > 默认兜底，**每个节点第一个命中的策略生效**。所以：

- `output.conv`（lm_head）和 `layers.*.feed_forward.w2_conv`（MLP down_proj）→ **16a8w per-channel**。
- 其余所有 conv2d（wq/wk/wv/wo、gate/up proj）→ **16a4w per-block**。
- 非 conv2d 的算子（add、mul、rmsnorm 等）→ 默认 **16a4w per-tensor**。

**为什么单独保护这两层**（recipe docstring `:646-657`）：down_proj 直接写回残差流，是经典的 outlier-heavy 层；而 DFlash 草稿正是消费残差流的隐藏状态（第 1/9/17/25/33 层），保护 down_proj 同时改善 4B 精度和 draft 接受率。lm_head 直接决定 logits，也要保精度。

---

## 4. 各粒度的具体语义

`QuantGranularity`：`PER_TENSOR=0, PER_CHANNEL=1, PER_BLOCK=2`（`quant_recipe.py:35-46`）。

### 4.1 per-tensor 激活（非对称）
整张激活张量共用一个 `scale` 和一个 `zero_point`。`MinMaxObserver` 在校准时统计全张量的 min/max。**非对称**意味着 zero_point 非零 —— 这也是为什么 `<|im_start|>` 这类离群值会把 scale 拉大、压垮正常小值（per-tensor 的固有代价）。

### 4.2 per-block 权重（LPBQ，关键细节）
`block_size=(1,16,1,1)` 作用在 conv2d 权重的 OIHW 布局上：dim 1 = 输入通道，**每 16 个输入通道一组**，每组（对每个输出通道）一个 fp32 scale。即 scale 张量形状 = `[out_ch, in_ch/16]`。

但它不是普通逐块量化，而是 **LPBQ（blockwise expansion，`QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION`）**（`builders/node_visitor.py:151-219`、`observers/per_block_param_observer.py:42-43`）：

> **两级量化**——每组的 fp32 scale 本身又被量化成 **4-bit**（`num_steps=16`），外加每个输出通道一个 fp32 的 `max_scale`。即：权重是 int4，scale 也是 int4（乘一个 per-channel 的 fp32 顶格 scale）。

这是为了在保留逐块精度的同时，把 scale 的存储也压下来 —— 纯 PyTorch 复现时这一步最容易被漏掉。

### 4.3 per-channel 权重（对称）
lm_head / down_proj 用逐输出通道一个 scale，对称，int8 容器，观测器 `PerChannelParamObserver`。

---

## 5. 特殊处理

### 5.1 KV cache → 8-bit（`annotate_kv_8bit`）
`backends/qualcomm/quantizer/custom_annotation.py:94-276`。对每个 `aten.matmul.default`（Q·Kᵀ 和 attn·V）：

- matmul 本身按 **16a8w** 标注（act uint16 / weight int8）；
- 沿 K/V 生产链（select/slice/permute/view/cat/rms_norm/stack …）向上标 **8a8w 对称 uint8**，K 路径标到 RoPE 的 add/sub 为止，V 路径标到 conv2d 为止。

净效果：**KV cache 是 8-bit uint8 对称**，这是内存和带宽的大头，压到 8-bit 收益最大。

### 5.2 词嵌入表：量化成 uint16 16-bit per-tensor（关键：它被当"激活"而非"权重"量化）

**结论先行（三方证据闭环：recipe 标注表 + QDQ 图 + 标注器代码，两次编译 b16-DFlash-4B 与 qwen3-1.7B-KV 一致）：词嵌入表被量化成 uint16（16-bit）per-tensor，它的输出激活也是 uint16 per-tensor。**

**机制（为什么是 16-bit 而不是 `4w` 的 int4）：**
- recipe 标注表里 `tok_embeddings | aten.embedding.default | use_16a4w | PER_TENSOR` —— embedding 落**默认档**（它不是 conv2d，没被 recipe 的 `add_node_target`/`add_regex` 覆盖）。
- `use_16a4w` = `16a`(激活 uint16) + `4w`(权重 int4)，但 **`4w` 只对 conv/linear 生效**：只有 `annotate_conv`（`rules.py:200` `input_qspec_map[weight]=config.weight`）才把权重标成 int4。
- **`aten.embedding` 走默认标注器 `annotate_single_in_single_out`（`rules.py:132-142`）**，它把 `args[0]`（那张表）标成 `input_activation`（uint16），**根本不赋 weight qspec**。所以表拿到的是 `16a`(uint16)，不是 `4w`(int4)。
- 图里因此是 `_frozen_param0(int32,uint16 值)→ dequantize_per_tensor([0,65535], zp≈35836)→ aten.embedding`，输出再 `quantize_per_tensor`。

**为什么用 uint16 定点而不是 fp16（都是 16 位）：** HTP 是整数引擎,整张 16a4w 图跑的是定点(激活 uint16、权重 int4/int8、int32 累加)。embedding 表量化成 uint16 定点,gather 出来的行**直接就是定点激活**,无缝进流水线;若留 fp16 则要在 embedding→第一层插 fp16→定点转换、且需要 fp16 路径。精度上 uint16-per-tensor 与 fp16 对 embedding 都近乎无损,所以选定点(对齐硬件)。fp16 只出现在**图的 I/O 边界**(如 logits 出图),图内部计算全是定点。

**另外两条 embedding 量化路径(对 DFlash 都没启用,别混淆)：**
- `tok_embedding_quantizer`(16a8w per-channel,`llm_wrappers.py:656`)—— 被 `if self.apply_embedding:` 门控(:667/:690/:739),`apply_embedding` 默认 False,只有**多模态**(:1224)才 True。DFlash 显式 False(`dflash_wrappers.py:712`)。
- `get_quant_embedding_transform`(CPU 4-bit embedding,`:260-263`)—— 只在传 `--embedding_quantize`(默认 None)时生效,DFlash 没传。

这两条是"给 embedding 换更激进量化/换 CPU 查表"的**可选**路径,关掉不影响上面那条**默认档的 uint16 量化**——词嵌入表**始终被默认档量化成 uint16**。

> 订正记录：这一段我先后写错过两次——先漏了默认档(误以为 16a8w per-channel),又矫枉过正说"不量化 fp16"。以上是用 recipe 标注表 + QDQ 图 + 标注器代码坐实的最终版。主机 fake-quant harness 里把 `embed_tokens` 留 fp32、不量化,和设备的 uint16-per-tensor **精度上几乎等价**(都 16-bit,对 embedding 近无损),所以 host accept 结果不受影响。

### 5.3 r3 旋转（SpinQuant R3）
Qwen3-4B `r3=True`（`__init__.py:545`）。在 Q/K 上（RoPE 前）左乘一个 **Hadamard 矩阵**（`scipy.linalg.hadamard(head_dim)` 归一化，`model/static_llama.py:96-100, 422-424, 490-492`）。作用是把注意力里的离群值“旋转摊平”，让 per-tensor/per-channel 量化更好量。这是一个**等价变换**（Hadamard 正交），不改数学结果，只改数值分布。

### 5.4 logits → 16-bit 出图
`get_logits_output_bit_width()` 恒返回 16（`static_llm_quant_recipe.py:44-46`）。logits 以 uint16 出图，在图边界 dequant 成 fp16（HTP 无 fp32）。

---

## 6. 量化后在图里长什么样（QDQ 表示）

`convert_pt2e` 会把每个算子包成 `dequantize → op → quantize` 三元组（QDQ）。QNN 后端识别这些 QDQ 算子并把它们融成一个整数 kernel。图里出现的 dequant 算子按粒度不同：

| 粒度 | 量化/反量化算子 |
|---|---|
| per-tensor 激活 | `quantized_decomposed.quantize_per_tensor` / `dequantize_per_tensor`（`node_visitor.py:82-86`） |
| per-channel 权重（w2/output） | `quantized_decomposed.quantize_per_channel` / `dequantize_per_channel`（`:77-80`） |
| **per-block 权重（16a4w_block）** | **`torchao.quantize_affine` / `torchao.dequantize_affine`**（`:307-312`），**不是** `dequantize_per_channel_group` |

> **重要**：`dq → matmul → q` 只是数学上的**参考语义**。真实 HTP 上，QNN 把这一串融成**一个定点整数 kernel**：int16 激活 × int4 权重 → int32 累加 → 定点 requantize，中间**不会真的走一遍 fp16**。int32 容器在编码时被压回 uint16（`node_visitor.py:335-336`），4-bit 值被 mask 成一个 nibble（`:338-340`）。

QDQ 后的图会存成 `decode_qdq.pt2`（`wrappers/llm_wrappers.py:731-737`），这是一个**可以直接在 CPU 上前向的 fake-quant `GraphModule`** —— 见下面的问题一。

---

## 7. 全流程（静态 PTQ）

`wrappers/llm_wrappers.py:605-787`：

```
torch.export.export(decoder)                          # :675  导出 fp32 图
  ↓
make_quantizer() + set_recipe(Qwen3_4BQuantRecipe)    # :653-654
  ↓
prepare_pt2e(decoder, quantizer)                      # :689  插入 observer
  ↓
_calibrate(...)  ← 用真实数据集（lm_eval 任务/用户 prompt）跑前向   # :705-712
  ↓
convert_pt2e(decoder)                                 # :720  observer → 固定 scale/zp 的 QDQ
  ↓
（存 decode_qdq.pt2） → TagQuantIO 打 IO dtype → to_backend（QNN lower 成 HTP 二进制）
```

- **观测器**：激活和 per-tensor 权重用 `MinMaxObserver`，per-channel 用 `PerChannelParamObserver`，per-block 用 `PerBlockParamObserver`。
- **静态 PTQ**：全程 `is_qat=False`；权重观测器只标定一次（有 `self.calibrated` 守卫）。**不是 QAT**，不需要重训。
- 校准跑在 **CPU**（`torch.set_num_threads` 自动调线程数，`decoder_utils.py:103` 的 device 默认 CPU）。

---

## 附：源码索引

| 内容 | 位置 |
|---|---|
| Qwen3-4B recipe | `examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py:646` |
| Qwen3-4B model config（r3、num_sharding） | `examples/qualcomm/oss_scripts/llama/__init__.py:528-548` |
| QuantDtype 枚举 + qconfig 映射 | `backends/qualcomm/quantizer/quantizer.py:80-178` |
| 各位宽的确切范围 | `backends/qualcomm/quantizer/qconfig.py`（16a4w: `:223-264`） |
| 粒度语义 / 策略优先级 | `backends/qualcomm/quantizer/quant_recipe.py:102-313` |
| KV 8-bit 标注 | `backends/qualcomm/quantizer/custom_annotation.py:94-276` |
| per-block LPBQ 观测器 | `backends/qualcomm/quantizer/observers/per_block_param_observer.py` |
| 图内 dequant 算子映射 | `backends/qualcomm/builders/node_visitor.py:77-86, 307-312` |
| PTQ 全流程 | `examples/qualcomm/oss_scripts/llama/wrappers/llm_wrappers.py:605-787` |
| Linear→Conv2d | `model/static_llama.py:348-366, 715`、`model/feed_forward.py:59-112` |
| r3 Hadamard | `model/static_llama.py:96-100, 422-424` |
| 设备端 GPU 后端 spec | `backends/qualcomm/utils/utils.py:976-1013` |
