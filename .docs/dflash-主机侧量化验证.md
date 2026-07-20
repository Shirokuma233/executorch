# DFlash 主机侧量化验证 harness

在**主机 CPU 上(无设备、无重编)**用纯 PyTorch fake-quant 复现 QNN 16a4w 量化,量化"量化对 DFlash accept_len 的影响"。用来在花代价上设备之前,先回答"量化 target / 量化 draft 各掉多少 accept"。

配套源码:`.docs/qwen3-4b-量化方法.md`(设备侧 16a4w recipe 的完整说明)。

---

## 1. 脚本(都在 `dflash/tools/`)

| 文件 | 作用 |
|---|---|
| `fake_quant_16a4w.py` | 核心 fake-quant 模块:`wrap_target_16a4w` / `wrap_draft_16a4w` / `HiddenQuant` / `Quant8Cache` / LPBQ / r3 |
| `dflash_quant_accept.py` | 底层驱动:`run_accept`(DeepSpec block7 贪心推测解码 loop)、`calibrate`、`encode` |
| `bench_host_accept.py` | **target 量化 vs fp32** 的 accept 对比(真实 benchmark,和设备同源 prompt) |
| `bench_host_draft_quant.py` | **4 档消融**(fp32 / 只量target / 只量draft / 都量),测草稿量化代价 |
| `diag_activation.py` | 诊断:证明激活确实量化了、LPBQ/r3 分档消融 |

**依赖**:DeepSpec(`Qwen3DSparkModel`,脚本用 `sys.path.insert` 引入 `../../DeepSpec`)、executorch 环境、`HF_HUB_OFFLINE=1`。模型:HF `Qwen/Qwen3-4B` 当 target(fp32),`dflash/draft_models/dflash_qwen3_4b_block7` 当 draft。**纯 host,不碰设备。**

## 2. 怎么跑

```bash
PY=/home/zqchen/tool/miniconda3/envs/executorch/bin/python

# target 量化 vs fp32(5 benchmark × 20 条)
HF_HUB_OFFLINE=1 MAX_NEW=128 N_EVAL=20 BENCH=all THREADS=16 \
  $PY dflash/tools/bench_host_accept.py

# 草稿量化 4 档消融(3 benchmark × 15 条)
HF_HUB_OFFLINE=1 MAX_NEW=96 N_EVAL=15 BENCH=gsm8k,math500,humaneval THREADS=16 \
  $PY dflash/tools/bench_host_draft_quant.py

# 诊断(激活量化证据 + LPBQ/r3 消融)
HF_HUB_OFFLINE=1 MAX_NEW=64 N_EVAL=5 THREADS=16 \
  $PY dflash/tools/diag_activation.py
```

env:`MAX_NEW`(每条生成上限)、`N_EVAL`(每 benchmark 条数)、`N_CALIB_PER`(每 benchmark 校准条数)、`BENCH`、`THREADS`。结果存 `benchmark/host_accept/` 和 `benchmark/host_draft_quant/` 的 `summary.json`。CPU fp32 跑 4B,一条约 4–10s(MAX_NEW=96),全量约 1 小时,后台跑。

## 3. fake-quant 复现的量化方案

严格照设备 recipe(细节见 `.docs/qwen3-4b-量化方法.md`):

**Target(`wrap_target_16a4w` + `HiddenQuant` + `Quant8Cache`)**
- 权重:q/k/v/o、gate/up → int4 per-group-16 对称;**lm_head、down_proj → int8 per-channel**
- 激活:uint16 per-tensor 非对称,MinMax 校准
- KV cache:uint8 per-tensor(`Quant8Cache` 子类化 `DynamicCache`,写入时量化)
- 喂草稿的 5 个隐藏状态:uint16 per-tensor(`HiddenQuant`,**带离群值**)
- **LPBQ**:per-group scale 再 4-bit 量化;**r3**:K 在 Hadamard 基里做 8-bit(等价设备 SpinQuant R3)

**Draft(`wrap_draft_16a4w`)**
- 权重:一般 int4 per-group;**fc、lm_head、down_proj → int8 per-channel**(fc 带 ~281k attention-sink 离群值,最敏感)
- 激活:uint16 per-tensor
- `embed_tokens` 是 Embedding,不量化(noise 查表保持 fp16,= 设备 embed.bin)

**未建模(二阶/无法 host 复现,已注明)**:注意力 QK·V 的 16a8w、真实定点 requantize、embed/RoPE-freqs 量化。fp16 fake-quant 与设备定点非 bit-exact,但**量化方法一致**,而这决定 accept。

## 4. 校准方法

只有**激活**要校准(权重是静态的,直接从权重算 min/max)。激活 per-tensor uint16 需要 `scale=(max-min)/65535, zp=round(-min/scale)`,靠 MinMaxObserver 跑前向收集——就是设备的静态 PTQ。

- **Target**:在校准 prompt 上跑普通 token 前向(`output_hidden_states=True`),各 QuantLinear 记输入激活 min/max,5 个 hidden 输出也记 → freeze。
- **Draft**(不同,因为输入是 target_hidden 不是 token):
  1. 先把 target + hidden 量化**校准并冻结**;
  2. 用**量化后的 target** 跑真实 accept loop,草稿 QuantLinear(校准模式)在它**部署时真正会见到的量化-target hidden 分布**上收集 min/max;
  3. freeze 草稿。

校准集用各 benchmark seed-42 shuffle 的尾部 `[N_EVAL:N_EVAL+N_CALIB_PER]`,**不与评测 prompt 重叠**。

## 5. 结果

### 5.1 Target 量化(`bench_host_accept.py`,N=20,全对齐 LPBQ+r3)

| benchmark | fp32 | 16a4w | 保留 |
|---|---|---|---|
| gsm8k | 5.80 | 5.80 | 99.9% |
| math500 | 6.03 | 5.71 | 94.6% |
| humaneval | 5.00 | 5.10 | 102% |
| mbpp | 4.97 | 4.87 | 98.0% |
| mt-bench | 4.47 | 4.39 | 98.2% |
| **均值** | **5.25** | **5.17** | **~98.5%** |

**结论:target 16a4w 很温和,平均掉 ~1.5%**。唯一有真实损失的是 math500(数学推理对精度最敏感)。隐藏 scale 每次复现设备:**L1=0.0015(干净),L9–33≈0.32(被离群值钉住)**。

### 5.2 Draft 量化 4 档消融(`bench_host_draft_quant.py`,N=15)

| bench | fp32 | T(只量target) | D(只量draft) | T+D(都量) | T+D/fp32 |
|---|---|---|---|---|---|
| gsm8k | 5.66 | 5.57 | 5.61 | 5.68 | 100.4% |
| math500 | 6.05 | 5.82 | 5.90 | 5.78 | 95.6% |
| humaneval | 5.01 | 5.07 | 5.06 | 5.01 | 100.1% |

**结论:量化草稿几乎免费**。
- **D vs fp32(只量草稿)**:−0.05 / −0.15 / +0.05 → 噪声内,≈0。
- **T+D vs T(已量化 target 上再量草稿的增量)**:≈0。唯一损失仍是 math500 上 **target** 的量化,草稿在此之上不加。

为什么草稿这么扛量化:fc/lm_head int8 保护;fc 的输入(target_hidden)本来就被 target 量化成 uint16 了,再量增量小;草稿主要靠 target_hidden 强信号,自己权重变 int4 影响不大。

> 诚实边界:N=15、host fp32 参考、非设备 bit-exact;per-prompt 噪声约 ±0.6,小于表里多数差值,所以严格结论是"target 温和、draft 几乎零成本",不是精确的百分位。

## 6. 设备端草稿 PTQ 待办(host 已证明值得做)

草稿现在是 fp16(`DFlashDraftCompiler.quantize` 是空壳 "no PTQ";`DFLASH_DRAFT: skip_quantize=True`)。host 实测量化它 accept ≈0 掉,收益 ~1.9GB + HTP 提速 + 根治 fp16 动态范围溢出。要上设备需:
1. 把 `dflash_wrappers.py` 的 `DFlashDraftCompiler.quantize` 从空壳补成真 PTQ(`prepare_pt2e → 校准 → convert_pt2e`);
2. 写草稿 quant recipe(5 层 + fc 5H→H + 非因果块注意力;target 16a4w 大体可套,fc 需保护);
3. `skip_quantize=False`;校准要喂 **target_hidden**(先跑 target)。

## 7. 相关:emb/lm_head 权重是同一张表(去冗余)

验证过 `target.embed == target.lm_head(Qwen3-4B tied) == draft.embed == draft.lm_head`,逐 bit 相同,都是 `[151936,2560]`。设备上这张 ~777MB 表存了 2–3 份(target pte 量化版、`embed.bin` fp16、block7 draft pte 的 in-graph lm_head fp16)。

去冗余最省事的一步:草稿从 `dflash_block7_lmh` 换成 `dflash_block7`(无 in-graph lm_head)→ 草稿 pte 不存表,noise-embed 和 lm_head 都复用那份 `embed.bin`(代价:lm_head 从 HTP 挪 CPU host scan)。跨 target/draft 做不到 1 份(target 要量化在 HTP、host 要 fp16,且两个 pte 是独立 QNN context,`use_weight_sharing` 不跨 context)。
