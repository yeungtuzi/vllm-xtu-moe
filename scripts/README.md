# scripts 索引

构建、基准、数值验证与端到端测试脚本。多数脚本用环境变量指定模型路径/规模,
直接运行会打印用法。

## 构建

| 脚本 | 作用 |
|---|---|
| `build_engine_variants.sh` | 构建内置 CPU 引擎的全部 ISA 变体(5 个 `.so`) |
| `check_upstream_drift.sh` | 检查上游 vLLM 变动对本项目补丁的影响(`REPO=<vllm 目录>`) |

## 基准

| 脚本 | 作用 |
|---|---|
| `bench_cpu_engine.py` | CPU 引擎吞吐(真实路由;`XIAOTU_LAYER1_NPZ` 指定真实层 fixture) |
| `bench_cpu_engine_bf16.py` | 同上,BF16 权重路径对照 |
| `bench_fp8_engine.py` | FP8 引擎吞吐(`E H I topk` 可选参数) |
| `bench_llm.py` | 端到端 decode 吞吐(`TEST_MODEL`、`CONCURRENCY`) |
| `bench_gpu_moe_prefetch.py` | 长 prefill 逐层 GPU 流式:同步 vs 重叠 |
| `dsv4_prefill_curve.py` | 端到端 TTFT / 并发 / decode 曲线 |
| `server_concurrency_test.py` | 已启动服务的并发吞吐测试 |

## 数值正确性

| 脚本 | 作用 |
|---|---|
| `test_swiglu_clamp.py` | 自包含:引擎 gated 激活(含 `swiglu_limit/alpha/beta`)vs torch 参考 |
| `test_swiglu_clamp_mxfp4.py` | 同上,MXFP4 路径 + 真实权重 fixture(`XIAOTU_LAYER1_NPZ`) |
| `test_glm53_fp8_layer.py` | 真实 GLM-5.3 fp8 专家层 vs numpy 参考(`GLM_MODEL`) |
| `gpu_prefill_golden.py` | 长 prefill GPU 内核 vs torch 参考 |

## 集成 / 端到端

| 脚本 | 作用 |
|---|---|
| `probe_oracle.py` | 不加载权重,探测各格式会被选中哪个后端 |
| `make_tiny_mixtral.py` | 生成 vLLM 命名兼容的微型 Mixtral(`TINY_SRC` 提供 tokenizer 模板) |
| `tiny_moe_equiv.py` | CPU 专家 vs GPU 专家的端到端数值等价性(`TINY_MODEL`) |
| `fp8_moe_smoke.py` | 真实 fp8 MoE 模型端到端冒烟(`SMOKE_MODEL`) |
| `fp8_equiv.py` | 同一 prompt 下 CPU/GPU 专家的 prompt-logprob 对照(`SMOKE_MODEL`) |
