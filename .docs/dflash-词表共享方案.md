# DFlash 词表(emb + lm_head)共享方案

> **状态:设计方案,未实现。** 目标是把设备上重复 2–4 份的 `[151936, 2560]` 词表,收敛成 **emb 一份(uint16)+ lm_head 一份(int8)、各保留原量化**,由 target 和 draft 共调。(emb 和 lm_head 量化方式不同、本就是两份不同数据,不强行合成一份——见 §2/§4。)

相关:`.docs/qwen3-4b-量化方法.md`(设备侧 16a4w)、`.docs/dflash-主机侧量化验证.md`(host fake-quant 实测)。

---

## 1. 背景:同一张表被存了 2–4 份

已验证 `target.embed == target.lm_head(Qwen3-4B tied) == draft.embed == draft.lm_head`,逐 bit 相同。当前设备上的分布:

| # | 内容 | 位置 | 引擎 | 格式 |
|---|---|---|---|---|
| 1 | target embed | target pte 第一分片(`llama.fallback` 是分片标记,不是 CPU 回退) | HTP | **uint16 16-bit per-tensor(定点)**——走默认档、表被当激活量化,非 int4 |
| 2 | target lm_head(output.conv) | target pte 最后分片 | HTP | int8 量化 |
| 3 | 草稿 noise-embed | `embed.bin`(777MB) | **CPU / host** | fp16 |
| 4 | 草稿 lm_head(`_lmh` 变体) | draft pte | HTP | fp16 |

`[151936,2560]` 16-bit ≈ 778MB(target embed 是 uint16 定点,embed.bin/草稿 lm_head 是 fp16,都 2 字节/值)。设备上光这张表就占 ~2–2.7GB。

## 2. 核心思路:emb 和 lm_head 各做成一个独立 pte,各保留原量化,被 target/draft 共调

**关键认识:emb 和 lm_head 虽然源自同一张 fp 表,但被量化成了两份不同的数据(emb = uint16 per-tensor 当激活量化;lm_head = int8 per-channel 当权重量化;见 §4)——它俩之间没有可去的冗余,硬合成一份反而要重量化、掉精度。真正重复的是"跨模型"的拷贝(target 的 vs 草稿的)。**

所以方案是**两个独立的 pte,各自保留现有量化方式**:

- **`emb.pte`**:`token_ids [*, T] → vectors [*, T, 2560]`,**uint16 per-tensor**(= target 现在的 embed 量化,原样)。
- **`lm_head.pte`**:`hidden [*, T, 2560] → logits [*, T, 151936]`,**int8 per-channel**(= target 现在的 lm_head 量化,原样)。

**"共享" = runner 让 target 和 draft 都去调这同一个 emb.pte / lm_head.pte**(不是 QNN 跨 context 权重共享——那条路不通)。每张表在设备上**只存一份**(在各自 pte 里),target 和 draft 通过 runner 调用复用它。**不用重量化、不掉精度。**

去冗余效果:
- `emb.pte`(uint16,778MB)替掉 { target 内 embed(uint16) + `embed.bin`(fp16,给草稿 noise) } 两份。
- `lm_head.pte`(int8,389MB)替掉 { target 内 lm_head(int8) + 草稿 pte 内 lm_head(fp16,`_lmh` 变体) } 两份。
- 词表相关从 ~2.7GB → **~1.17GB(778+389),省 ~1.5GB**,且精度不变。

## 3. 架构:两个共享 pte(emb / lm_head)+ headless 解码器 + runner 编排

```
   emb.pte(uint16 表,一份)                 lm_head.pte(int8 表,一份)
        ▲   │ vectors                            ▲   │ logits
 token_ids  │                            uint16  │   │
        │   ▼                             hidden │   ▼
   ┌────────────────┐   hidden(uint16)   ┌────────────────┐
   │ target.pte     │───────────────────▶│    (verify)    │
   │ (headless,     │                    └────────────────┘
   │  只出 hidden)  │
   └────────────────┘        ┌────────────────┐
   ┌────────────────┐───────▶│   (propose)    │  两个 decoder 都:
   │ draft.pte      │ hidden └────────────────┘   token→emb.pte→自己→lm_head.pte
   │ (headless,量化)│                              runner 串起来
   └────────────────┘
```

- **`emb.pte` / `lm_head.pte`**:各一份表,target 和 draft 都调(runner 调用,非 QNN 权重共享)。
- **target.pte(headless)**:只跑 36 层 transformer。**不在图内 embed**(吃 emb.pte 的输出),**不在图内 lm_head**(出 uint16 hidden,交给 lm_head.pte)。target 现在已经出 hidden(DFlash 用),要去掉 output_conv、改吃预算 embedding —— **导出层改动**。
- **draft.pte(headless,量化)**:`dflash_block7`(无 in-graph lm_head)已接近 headless(只出 hidden、noise 外部),但**要量化成 16a4w**,让它的 hidden 也是 uint16(才能喂共享的 int8-uint16 lm_head.pte,见 §4)。
- **runner**:编排 `emb.pte → decoder → lm_head.pte` 的调用链,传 hidden/logits、管两套 KV cache、做 accept 循环(不变)。

## 4. 精度设计:各保留原量化,唯一约束是"喂 lm_head.pte 的 hidden 必须 uint16"

**两个 pte 各保留现有量化,不做任何统一 / 重量化 —— 所以不掉精度:**
| pte | 位宽 | 粒度 | 量化角色 | 就是 target 现在的 |
|---|---|---|---|---|
| `emb.pte` | **uint16(16-bit)** | **per-tensor** | 当**激活**量化(默认档) | embed(`dequantize_per_tensor`) |
| `lm_head.pte` | **int8(8-bit)** | **per-channel** | 当**权重**量化 | output.conv(`dequantize_per_channel`) |

> 为什么不合成一个 pte:emb 和 lm_head 是**两份不同的量化数据**(位宽 16 vs 8、粒度 per-tensor vs per-channel),它俩之间没冗余;硬合成一份要重量化,int8 会让 embed 从 16→8 bit(实测相对误差 0.01%→0.9%)。分开两个 pte 各用原量化,**零精度损失**,是更优解。emb 现在 uint16 per-tensor 对这张均匀的表已近无损(见"实测:emb 表 max/median 仅 3x,per-tensor 误差 0.01%"),没必要动。

**唯一的精度约束(= "草稿必须量化"的根因):`lm_head.pte` 的输入激活是 uint16 定点**,所以喂进去的 hidden 必须是 uint16:
- target 的最终 hidden 本来就是 **uint16**(16a4w 激活)→ 直接喂 ✓
- 草稿若还是 **fp16**,hidden 进不了 uint16 输入的 lm_head.pte → **必须把草稿量化成 16a4w**(让它的 hidden 也 uint16),或至少在图末把 hidden 量化成 uint16。草稿量化 host 实测 accept 几乎不掉(见 §8),所以可行。

**精度边界统一在 uint16 hidden 上**:target 和 draft 都在各自 headless 图末尾出 uint16 hidden,`lm_head.pte` 统一吃 uint16。`emb.pte` 输出的向量则作为两个 decoder 的第一层输入激活(uint16)。

## 5. 草稿量化 + 校准(用主模型吐出的 hidden)

共享 `lm_head.pte`(uint16 输入)**强制草稿量化**(草稿 hidden 要变 uint16)。这一步 host 已实测**几乎不掉 accept**(见 §8),所以可行。

**草稿量化方案(照 target 的 16a4w 做):**
- 权重:一般 int4 per-group;`fc`(带 ~281k attention-sink 离群值)、lm_head 路径 → int8 per-channel 保护。
- 激活:uint16 per-tensor。
- `embed_tokens` 不量化(噪声查表交给 `emb.pte`)。

**校准(关键,照你说的用主模型吐出的隐藏层):**
1. 先跑 target 的 PTQ,**dump 出它在 layer 1/9/17/25/33 的量化隐藏状态**(就是喂给草稿的那 5 层 hidden)。
2. 草稿的激活 MinMax 校准**在这些"量化-target 吐出的 hidden"分布上做** —— 因为草稿部署时吃的就是它们。
3. 这套校准逻辑 host harness 里已跑通(`dflash/tools/bench_host_draft_quant.py`:先冻结量化 target,再用它跑 accept loop,草稿在真实量化 hidden 上收集 min/max)。设备侧照搬这个顺序即可。

## 6. 需要改的三块

**A. 导出**
- 新建 `emb.pte`(token→向量,**uint16 per-tensor**,= target 现在的 embed 量化)和 `lm_head.pte`(hidden→logits,**int8 per-channel**,= target 现在的 output.conv 量化)。**各保留原量化,不重量化。注意不能复用多模态的 `apply_embedding=True` 那套**(它是给 audio/vision encoder 的,接线和形状假设都不适配文本,得自己写导出)。
- target 改 headless:去掉 output_conv、改吃预算 embedding、末尾出 uint16 hidden。
- draft 改量化 + headless(结构基本是现成的 `dflash_block7`,加 PTQ)。

**B. 量化**
- 把 `DFlashDraftCompiler.quantize` 从空壳补成真 PTQ(现在是 `skip_quantize=True`)。
- 加草稿 recipe(§5)。
- 校准喂 target dump 的 hidden(§5)。
- emb 表和 lm_head 表各按原量化(uint16 per-tensor / int8 per-channel)分别导出成 `emb.pte` / `lm_head.pte`。

**C. runner**
- 加 `emb.pte` / `lm_head.pte` 的加载 + 调用。
- 编排:prefill 阶段 `emb.pte(prompt) → target → (extract hidden) → emb.pte(draft noise) → draft`;decode 阶段每轮 `emb.pte(noise) → draft propose → lm_head.pte → target verify → lm_head.pte → accept`。
- hidden 跨图传递用共享缓冲(注意:draft 侧 shared_buffer 目前有 QNN 1003,见 [[dflash-shared-buffer-1003]],多 `emb.pte`/`lm_head.pte` 两个边界会不会触发要验证)。
- 去掉 `embed.bin` 的 CPU 查表(草稿 noise 改走 `emb.pte` on HTP)和 CPU lm_head scan(改走 `lm_head.pte` on HTP)。

## 7. 收益 vs 代价

**收益**
- 词表**从 2–4 份收敛成 1 份** —— 省多少取决于统一到哪种量化(见 §4):**int8 ≈ 389MB(省最多 ~1.5–2.3GB,但 embed 从 16→8 bit 掉精度)**;uint16 ≈ 778MB(省少些,但精度不降)。
- 草稿 lm_head 从 **CPU host scan(742MB 表在 CPU 扫,慢)→ HTP**,**顺带提速**。
- 草稿量化本身再省 ~1.9GB(权重 fp16→int4)。

**代价**
- **多了图边界**:每步多 embed / lm_head 两次 QNN execute;logits 151936 宽(每次 ~2–4MB 跨图 I/O)+ graph launch。这加每步延迟——但草稿侧是净赚(HTP 换 CPU)。
- **精度边界**:draft hidden 要在图末量化成 uint16(host 实测对 accept 无损)。
- **工程量**:导出(headless target/draft + `emb.pte` + `lm_head.pte`)+ 量化(草稿 PTQ)+ runner 编排,三块都要动。

## 8. 已验证的支撑(host 实测,`.docs/dflash-主机侧量化验证.md`)

- **草稿量化几乎不掉 accept**:D(只量草稿)vs fp32 ≈ 0;T+D vs T ≈ 0。所以"共享图强制草稿量化"这个前提**站得住**。
- **草稿校准用量化-target 的 hidden** 这套已在 host 跑通,隐藏 scale 复现设备(L1=0.0015、L9-33≈0.32)。
- target 16a4w 本身 ~98.5% 保留。

## 9. 分阶段落地(建议顺序)

1. **host 端已基本验证**:草稿量化(16a4w,含 int8 lm_head)几乎不掉 accept(§8),而 emb/lm_head 各保留原量化不变 —— 所以"共享 + 草稿量化"这条链的精度前提已成立,无需再单跑 vocab 量化实验。
2. **设备端补草稿 PTQ**(§6-B),先不拆 emb/lm_head —— 单独验证"草稿量化能上设备、accept 不掉"。
3. **建 `emb.pte` + `lm_head.pte` + headless target/draft + runner 编排**(§6-A/C),端到端。
4. 处理 shared_buffer / 图边界的性能与 1003 风险。

## 10. 开放问题 / 风险

- **图边界性能**:多两次 QNN execute(尤其 151936 宽 logits)对吞吐的实际影响,要实测。可能需要把 lm_head 的 logits 直接在图内接 argmax/top-k,减少跨图 I/O。
- **shared_buffer**:draft 侧 shared_buffer 已有 QNN 1003([[dflash-shared-buffer-1003]]),再加 `emb.pte`/`lm_head.pte` 两个边界会不会更糟,要验。
- **AR/shape:每个 pte 要两张 AR 图,但表仍一份(已想清)。** target 和 draft 都是 decode(AR=8)+ prefill(AR=32);QNN 图静态 shape,一张吃不了两种长度,所以 `emb.pte`/`lm_head.pte` 都要按 AR 各编一份 —— 和现在 decoder 的 `prefill_forward`+`kv_forward` 一样。**同一 pte 内两张 AR 图靠 `use_weight_sharing`(context 内共享,decoder 已在用的成熟模式)共享同一份表,所以表物理拷贝仍是 emb 一份 + lm_head 一份,不因两张图翻倍。** 细节:`emb.pte` 需 {AR=32 prompt-prefill, AR=8 decode/noise};`lm_head.pte` 需 {AR=8 decode},prefill 只需最后一个位置的 logits 采样第一个 token → 可只编 **AR=1**,省掉 32×151936 的浪费。
  - ⚠ 盯 within-pte 权重共享:qwen3-1.7B KV 编译日志里出现过 6 条 `wtshare_operation.cc:1057::ERROR:small.is_shared()`(decoder 内 prefill/kv 对某"small"张量共享失败,编译仍完成)。建 `emb.pte`/`lm_head.pte` 时要确认那份**大表本身**是真共享、没退化成两份。
- **草稿必须量化**带来的连锁:draft 的 KV、noise 路径都要跟着量化对齐。

---

## 11. 实现进度（本次 session 记录，分支 `dflash_memory`）

> 状态:**emb.pte 和 lm_head.pte 都能在 AoT 编译期干净拆出并数值验证(1.7B / 4B);拆分方案从"分开量化"改成了"联合校准"(= 一起量化,分开成 pte)。设备端 runtime(M5)还没做。**

### 11.1 最终采用的拆分方式:联合校准(不是图手术)

拆 emb / lm_head 有两种实现"复用整图量化参数"的路子:

- **图手术(路B,未采用)**:整图量化一次(`apply_output=True`),再在 fx 图上从 `hidden→lm_head` 边界切开成两个 pte。逐位保证等价,但 fx 手术 intricate、易错。边界节点结构已 dump 清楚(见 §11.4)。
- **✅ 联合校准(实际采用)**:保持"headless decoder + 独立 LmHead + 独立 TokenEmbedding"三个模块,但**量化时把三个一起 prepare,用一个组合模块 `_SplitEval`(tokens→emb→decoder→lm_head→logits)跑真实轨迹校准,再一起 convert**。这就是"一起量化,切分 pte 的时候分开",低风险,且实测能把拆分做到和整图生成一致。

**为什么必须联合校准(关键根因)**:量化校准(`kv_inference`)靠 `argmax(模型输出)` 自回归喂数据。**headless decoder 的输出是 hidden**,`argmax(hidden)` 得到 [0,dim) 的乱码 token → decoder 在"真实前缀 + 一段乱码轨迹"上校准 → transformer+scale 被污染 → 生成乱码。联合校准让 argmax 用**真 lm_head 的 logits → 真 token → 真 hidden**,每个 observer 都看真实分布。

### 11.2 已实现的代码改动（都在 `examples/qualcomm/oss_scripts/llama/`）

- **`model/embedding.py`**:加 `LmHead`(wraps decoder.output conv,hidden→logits);`TokenEmbedding` 复用。
- **`model/static_llama.py`**:`apply_output` kwarg;`LlamaModel.forward`/`LlamaModelWithoutEmbedding.forward` 在 `apply_output=False` 时输出 `self.norm(hidden)`(**注意:是 norm 之后的 hidden,norm 属于 decoder,lm_head 只有投影**)。
- **`wrappers/llm_wrappers.py`**(主):
  - `_SplitEval`(module 顶层):tokens→[emb]→decoder→[lm_head]→logits 组合模块,用于联合校准 + 验证。
  - `_get_model_instance`:emb-split 文本走 `LlamaModelWithoutEmbedding` + 从 `decoder.tok_embeddings` 建 `TokenEmbedding`;headless 建 `LmHead`。
  - **`quantize()`**:emb-split 时 `apply_embedding=True`;lm_head-split(`not apply_output`)时把 lm_head 也 `prepare_pt2e`(16a8w per-channel),用 `_SplitEval` 组合模块联合校准,再一起 `convert_pt2e`。(旧的 `_quantize_lm_head` 单独校准已废弃,方法仍在但不调。)
  - `_override_lm_head_input_scale`:compile 时把 lm_head 输入 scale 注入成 decoder 输出 hidden 的 scale(边界无损)。
  - `_tag_ios`:emb 输入 embeds 走 fp32 边界(guard `not apply_embedding`);tok_embedding 输出 fp32(gate on apply_output)。
  - `_verify_generate`:用 `_SplitEval` 跑真实 prompt 的 KV 生成(2 个 prompt),打日志文本 —— 验证拆分是否和整图生成一致。
  - `HybridTextDecoder.compile`:emb.pte / lm_head.pte 的 lower 块;emb→decoder 走 **fp32 边界(option a)**,decoder→lm_head 走 **uint16 边界 + scale 注入**。
- **`decoder_utils.py`**`kv_inference`:`is_multimodal` → `use_external_embedding = tok_embedding is not None`(放宽 gate,让文本 emb-split 也走 embedding 输入路径)。
- **`llama.py`**:`--headless_decoder`(拆 lm_head)、`--split_embedding`(拆 emb)、`--verify_split`(拆后跑真实生成验证)三个 flag;`compile_specs`/`pte_filenames` 接 `LM_HEAD`/`TOK_EMBEDDING`。
- **`decoder_constants.py`**:`LM_HEAD`、`LM_HEAD_GRAPH_NAMES`。

### 11.3 已验证的结果（1.7B kv，`--headless_decoder --split_embedding`）

- **能拆**:产出 `tok_embedding_qnn.pte`(594MB uint16 表)+ `lm_head_qnn.pte`(300MB int8 表)+ headless decoder(773MB,表都移除了 = 1666 − 594 − 300)。4B 同样能拆(742/374MB,~30-60min)。
- **两张 AR 图共享一份表**:hybrid 下 lm_head.pte = 304MB(=1 份表,非 2 份)—— within-pte weight sharing 生效。
- **emb 拆分逐位一致**(logit mae=0);**lm_head/全 headless** 用联合校准后生成连贯、和整图基本一致;边界 scale 注入 = 整图 S_in(0.002136)。
- ⚠ **验证方法学坑(重要)**:量化有 **run-to-run 随机性** —— `_auto_tune_calibration_threads` 每次按耗时自动选线程数,不同线程数 → 并行归约顺序不同 → observer min/max 微变 → 量化不同 → **两次 mono 编译生成的文本都不一样**。所以"拆开 vs 整图"的分开编译对比**被噪声污染**,单看某次"逐字一致"不算数。**正确做法:两边都加 `--calibration_num_threads 1`(单线程确定性)再对比。** 本次在跑这个确定性对比时被暂停。

### 11.4 lm_head 边界的确切图结构（若将来要做图手术）

1.7B 整图(`apply_output=True`)QDQ,hidden→logits:`rms_norm(hidden,fp[1,1,2048]) → q/dq(S_in=0.002136,zp=27848) → reshape[1,1,2048,1] → q/dq → transpose[1,2048,1,1] → q/dq → conv2d(lm_head,dequantize_per_channel权重) → q/dq(S_out=0.001214,zp=24035) → transpose → q/dq → squeeze[1,1,151936] → q/dq → output`。(reshape/transpose 来自 `convert_linear_to_conv2d`。)切点 = hidden 的第一个 quantize。临时 dump 工具:`_dump_lmhead_boundary`(env `DUMP_BOUNDARY=1`),用完要删。

### 11.5 下一步

1. **跑完确定性对比**(`--calibration_num_threads 1` 的 mono vs split,逐字对)—— 给出"拆分是否确定性等价"的定论。若一致 → 联合校准这版就够了;若差 → 残差是 lm_head 权重量化(16a8w vs recipe),再考虑图手术复用 recipe。
2. **清理**:删临时 `_dump_lmhead_boundary` + `DUMP_BOUNDARY` 调用;`_quantize_lm_head` 已死可删。
3. **4B 确认** → **M3 共享**(runner 让 target/draft 都调同一份 emb.pte/lm_head.pte)→ **M4 草稿 PTQ** → **M5 runtime C++**(emb→decoder→lm_head 编排链,最大剩余工作,未动)。

**环境**:必须用 `/home/zqchen/tool/miniconda3/envs/executorch/bin/python`(base python 缺 torchao)。
**dev 编译命令**:`HF_HUB_OFFLINE=1 <上面的python> examples/qualcomm/oss_scripts/llama/llama.py --decoder_model qwen3-1_7b --model_mode kv -m SM8750 -b build-android -c -a <dir> --max_seq_len 1024 --prompt "Hello" --headless_decoder --split_embedding --calibration_num_threads 1 --verify_split`

---

## 12. DFlash 完整共享架构 + 量化定稿(本次 session 续,2026-07-20)

> 词表拆分(§11)已在通用 kv 路径验证。本节把它落到 **DFlash 模式**,并合入草稿量化,形成完整共享架构。**各图的 ar 数值由用户确定,本节只记结构、不写死。**

### 12.1 M4 草稿 PTQ:已实现 + 编译验证通过

- `DFlashDraftCompiler.quantize` 从 stub 补成真 PTQ(`dflash_wrappers.py`):
  - **数据流**:`DFlashManager.__init__` 把量化后的 target 引用交给 draft(`self.draft_compiler.target = self.target`;chain 顺序保证 target 先 `convert_pt2e`)。
  - **dump hidden**(`_dump_target_hidden`):用量化后的 `self.target.prefill.decoder` 对真实校准文本前向,取输出末尾 Lc 个 captured 拼成 `new_context`。因 target 走路 B 联合校准(§11.1),这 hidden 是整图完整质量。
  - **校准**(`_calibrate_draft`):照 runner `run_draft`(`dflash_token_generator.cpp:496-722`)布局逐 block 造 6 输入(noise=emb 查 `[committed,MASK...]` / 双向非因果 mask 三区 / new_context / 位置 / 自累积 past_kv)喂 prepared draft 收激活。
  - **recipe**(`DFlashDraftQuantRecipe`,`static_llm_quant_recipe.py`):default `use_16a4w_block` + `add_regex {fc.conv, layers.*.mlp.down_proj.conv, lm_head.conv}` → `use_16a8w` per-channel。
- **编译验证**:`--model_mode dflash --num_sharding 3`(默认 sharding=4 会撞 capture 层 18 的边界,必须改 3);不能 `--use_fp16`(draft PTQ 需量化后 target)。产物 `compiled_pte/qwen3-4B_dflash_ptq/`:`dflash_draft_htp.pte`=**779MB**(16a4w)vs fp16 1858MB → 省 **1.08GB**。
- **顺带修复**:headless target 的 lm_head split 逻辑条件化 —— `llm_wrappers.py:1234` 改 `if not self.apply_output and LM_HEAD in request.method_data`;DFlash 不传 LM_HEAD 就跳过(保持 v1.0 runner host-scan),通用词表拆分路径照传照拆、不受影响。
- ⚠ **当前 draft 仍含 in-graph lm_head**(ckpt `dflash_qwen3_4b_block7` 带 `lm_head.weight` → `with_lm_head=True`),下一步要去掉、共享 target 的 lm_head。

### 12.2 目标架构:四个模块,各一份,target/draft 共享

| 模块 | 输入 → 输出 | 谁用 |
|---|---|---|
| **emb.pte** | token_ids → embeds(uint16 表) | target 输入 + draft noise embed,**共享** |
| **headless target decoder** | embeds → final hidden + 选中层 hidden | target |
| **lm_head.pte** | hidden → logits(int8 per-channel) | target verify + draft propose,**共享** |
| **draft**(去 emb+lm_head) | target hidden → block hidden | draft |

draft 的 emb 本就没有(noise 是外部输入);这次把 **in-graph lm_head 也踢掉**(`with_lm_head=False`),draft 只出 hidden,靠共享 `lm_head.pte` 算 logits。

### 12.3 边界编码设计(**改掉 §11.3 的 fp32 option-a**)

| 路径 | 编码 | 机制 |
|---|---|---|
| emb → decoder → lm_head(主路径) | **全 uint16** | = 整图不切分时激活本来的走法;边界只传 scale、无损直传,**不再 dequant 成 fp32**(§11.3 的 emb→decoder fp32 是"先跑通"妥协,现改掉) |
| decoder 选中层 → draft | **fp16** | 最早 dflash 的设计;fp16/uint16 同 16bit,但 fp16 省掉"传 5 层不同 scale";draft **到 fc 输入才量化**(fp16→uint16,scale 是 draft 自己校准的,不用 target 传) |
| draft 出口 hidden → lm_head | **uint16** | scale 注入 = `get_logits_scale`,和 target hidden 同编码,才能喂同一个共享 lm_head(把 M1b 的 lm_head 输入注入,推广到 draft 输出侧) |

要点:主路径全 uint16(更省带宽 + 省 fp32 边界的 requant 误差),代价是要写 uint16 的 runtime;给 draft 的 hidden 走 fp16(绕开选中层 5 层 multi-scale);draft 出口再回 uint16 对齐共享 lm_head。

### 12.4 各图 ar(**结构;数值由用户确定**)

- **target**:prefill 图 + verify(kv)图,两张。
- **draft**:prefill 图 + decode(kv)图,两张。
- **emb.pte / lm_head.pte**:ar 不自定,被上游拽着走,要覆盖 target + draft 的多种 ar(比 target 独享时 ar 种类更多)。
- target verify 和 draft decode 的 ar 有 shifted 错位。
- **具体 ar 数值由用户确定,本文档不写死。**

### 12.5 host 验证方式(四模块 QDQ 存盘,验"整套输出正不正常")

- QNN `.pte` 是 HTP 设备图,host 跑不了;host 验证用 `convert_pt2e` 的 **QDQ GraphModule**(带同样量化 scale、CPU 能跑,数值 == pte)。
- **把 emb / decoder / lm_head / draft 四个 QDQ 图 save 下来**,host 加载跑完整 DFlash accept loop:`tokens→emb→decoder→(lm_head)→verify` + `hidden→draft→(lm_head)→propose→accept`。
- **判据**:量化整套流程**输出正不正常** = 生成文本连贯 + accept_len 合理(>1);**不纠结** QDQ-vs-fp32 的单步 top1 —— 量化必然有误差,15 万 vocab 上小噪声就翻 argmax,top1 低不代表崩(实测某次 top1=0.14 但 cos=0.90,是度量选错、非量化坏)。
- QDQ 存盘后,验证/迭代直接加载复用,**不重编**。

### 12.6 待办(顺序)

1. **主模型词表拆分搬到 DFlash**:`DFlashManager` 配 `TOK_EMBEDDING`+`LM_HEAD` 的 method_data / compile_specs / pte_filenames(照 `MultiModalManager`),target 设 `apply_embedding=True`+`apply_output=False` → 产 `emb.pte` + headless decoder + `lm_head.pte`。
2. **draft 去 in-graph lm_head**:`with_lm_head=False`,出口 hidden(uint16,scale 注入 `get_logits_scale`)共享 `lm_head.pte`。
3. **主路径改全 uint16 边界**:emb→decoder 从 fp32 改 uint16 + scale 注入(§12.3)。
4. **四模块 QDQ 存盘 + host 端到端 accept 验证**(§12.5)。
5. **M5 runtime C++**:uint16 边界的 emb→decoder→lm_head + draft 编排链。
