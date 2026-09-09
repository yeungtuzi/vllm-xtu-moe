# 通用化路线图

目标:让本插件对**任意 MoE 模型**(DeepSeek / GLM / Qwen / Mixtral / …)
和**任意 x86 指令集**(AVX2 → AVX-512 → AMX)都可用。

---

## 0. 现状

| 维度 | 现状 |
|---|---|
| **指令集** | 引擎有 5 级运行时阶梯(`scalar → avx2 → avx512_base → avx512_vnni → avx512_bf16`),`scripts/build_engine_variants.sh` 一次编出全部变体;AMX 未接线(需要 AMX 硬件验证) |
| **数据格式** | BF16 / FP16 / FP8(e4m3 W8A16)/ MXFP4 / NVFP4 / WNA16 已有内核;INT4 组量化与零点的适配进行中;INT8、MXFP8、GGUF k-quants 未做 |
| **模型通用性** | ✅ 已从"单模型覆盖"升级为**通用 CPU experts 后端**:按格式注册进 `FusedMoEFactory`,任意模型自动选中 |
| **路由** | ✅ 复用主线 router(softmax / sigmoid+noaux_tc / sqrtsoftplus / grouped / custom) |
| **激活** | ✅ packed gated 激活 + `swiglu_limit/alpha/beta`;交错布局(`SWIGLUOAI`)明确拒绝 |
| **测试** | 引擎侧有数值 golden(真实权重);缺"格式 × ISA"矩阵 |

## 1. 指令集

| 任务 | 内容 | 量级 |
|---|---|---|
| **打包全部 ISA** | 已提供 `scripts/build_engine_variants.sh`;还需把 5 个 `.so` 纳入发布 wheel,并加一个启动自检 | 0.5–1 天 |
| **AMX 接线** | 引擎里已有 ggml 风格的 AMX tile GEMM 源码,但未接入;需要预 pack 权重、增加 `avx512_amx` 变体、在 Intel SPR/EMR 上验证 | 1–2 周 |
| **AVX-512 FP16 / VNNI 细化** | 目前 vnni 变体已编译但内核未专门使用 int8 路径 | 3–5 天 |

> AMX 只能加速 bf16 / int8 两类 tile 运算;MXFP4 走 AMX 需要先反量化到 bf16。
> 实测表明当前 CPU 引擎**不是带宽受限**(权重流量远低于内存带宽),因此
> AMX 的收益需要实测确认后再投入。

## 2. 数据格式

| 格式 | 谁需要 | 优先级 |
|---|---|---|
| **INT4 组量化(含零点)** | GPTQ / AWQ 系模型 | **P1** |
| **INT8 W8A8** | 常见服务端量化 | P1 |
| **MXFP8 / MXFP6** | 新一代 MX 格式 | P2 |
| **FP8 e5m2 / FNUZ** | ROCm 生态 | P2 |
| **GGUF k-quants(Q4_K/Q5_K/Q6_K)** | llama.cpp 生态 | P3 |

新增一个格式的固定成本:实现 traits(4 个方法)+ 权重映射 + golden 测试 + 文档。

## 3. 后端能力

| 任务 | 说明 | 优先级 |
|---|---|---|
| ~~专家并行(expert_map / EP)~~ | ✅ 代码已支持(映射全局→本地 id);端到端验证受限于本机 TP=2 卡死 | – |
| **FP8 内核提速** | 🟡 已完成两步:①去 LUT gather(位运算解码),吞吐 0.35→0.98 TFLOP/s;②小批量 N-切片 + worker 子集,B=1 单层 7.1→**0.745 ms**、端到端 decode 5.8→**11.6 tok/s**。下一步:权重预转 bf16 镜像(去掉解码算术,预期再 3–5×)与分块 GEMM(权重行复用 M 个 token) | P1 |
| **INT4 真实组大小/零点** | 目前组大小写死 128 | P1 |
| **交错 gate/up 布局** | 让 gpt-oss 系(`SWIGLUOAI`)可用 | P2 |
| **`apply_router_weight_on_input`** | 少数模型使用 | P3 |

## 4. 测试与发布

- **格式 × ISA 矩阵**:每个格式在每个 ISA 变体上都有数值 golden;
- **端到端等价性**:CPU 专家 vs GPU 专家的 prompt-logprob 对照(已有
  `scripts/tiny_moe_equiv.py`);
- **发布**:一个 wheel 内含全部 ISA 变体(每个约 1.2 MB);
- **自检**:启动时打印 CPU flags、选中的变体、可用格式。

## 5. 建议顺序

| 顺序 | 事项 | 为什么 |
|---|---|---|
| 1 | 打包全部 ISA 变体 | 成本最低,立刻让 AVX2-only 机器可用 |
| 2 | 专家并行 + FP8 提速 | 直接决定真实模型的可用性与速度 |
| 3 | INT4 组量化 / INT8 | 对齐主线 CPU 的格式覆盖面 |
| 4 | 格式 × ISA 测试矩阵 + 自检 | 防止回归 |
| 5 | AMX 接线 | 上限最高,但需要硬件验证 |
| 6 | 长尾格式(MXFP8 / GGUF / …) | 按需求增量 |
