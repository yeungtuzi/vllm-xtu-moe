# xiaotu-moe ↔ Lvllmds4-x 集成（drop-in 替换 lk_moe）

作者：agent（DeepSeek-V4-Flash-0731）· 阶段：Phase 4 集成设计（已核实，**落地与全量层回归留给"测试最后统一做"**）

## 目标
把 `Lvllm`/`Lvllmds4-x` 里 `self.lk_moe`（闭源 PyPI lk_moe）替换为 `xiaotu_moe.load()` 实例，接 `_cpu_decode`/`_cpu_prefill`，全量 MoE 层数值/性能回归，证明 xiaotu-moe 是真实可用的 drop-in。

## 为什么只需极小的 fork 改动
xiaotu-moe 与 lk_moe 的**构造签名 1:1 兼容、方法名一致**（`MOEConfigV2` 19 字段、`(cfg, w13, w2, w13_scale, w2_scale, w13_gs, w2_gs)` 全整型 data_ptr、`cpu_decode(stream,qlen,k,hidden,ids,weights,out)`、`cpu_prefill(qlen,k,ids,weights,in,out)`、`gpu_prefill(...)`）。`moe_runner.py` 的三重分派无需任何改动。

唯一差异是**数据通道**：
- `_cpu_prefill`（L1726）已把 hidden/ids/weights 拷到 CPU 再算 → xiaotu-moe 直接可用 ✅
- `_cpu_decode`（L1708）在 CUDA-graph capture 时把 **GPU 指针** 直接传给引擎（闭源 lk_moe 靠自身 CUDA 读/写）。xiaotu-moe 纯 CPU 不能读 GPU 地址 → **唯一需要桥接的点**。

## 桥接方案（`integration/routed_experts.cpu_decode.patch`，已 `git apply --check` 通过）
把 `_cpu_decode` 改成与 `_cpu_prefill` 相同的模式，但把 CPU 结果写回**固定的 `RoutedExperts.output_gpu` GPU 缓冲**（graph capture 要求输出地址稳定）：

```python
def _cpu_decode(self, hidden_states, topk_weights, topk_ids):
    qlen = hidden_states.size(0)
    expert_ids_cpu = topk_ids.to(dtype=torch.int32, device='cpu', non_blocking=True)
    weights_cpu = topk_weights.to(dtype=torch.float32, device='cpu', non_blocking=True)
    hidden_states_cpu = hidden_states.to(device='cpu', non_blocking=True)
    output_cpu = torch.empty((qlen, self.hidden_size), dtype=torch.float32, device='cpu')
    torch.cuda.current_stream().synchronize()
    self.lk_moe.cpu_prefill(qlen, self.top_k,
        expert_ids_cpu.data_ptr(), weights_cpu.data_ptr(),
        hidden_states_cpu.data_ptr(), output_cpu.data_ptr())
    RoutedExperts.output_gpu[:qlen].copy_(output_cpu, non_blocking=True)
    output = RoutedExperts.output_gpu[:qlen]
    if self.check_nan_in_output:
        torch.nan_to_num(output, nan=0.0, out=output)
    return output.to(hidden_states.dtype)
```

- `RoutedExperts.output_gpu` 是 `_initialize_cuda_graph_buffers` 预分配的 `(max_num_seqs, hidden)` fp32 GPU 缓冲（graph 安全）。
- `.to(device='cpu')`/`torch.empty` CPU 是不进 graph 的 host 算子；`output_gpu.copy_` 是 graph 捕获的 GPU 写、地址稳定。
- **零闭源复制**：只是把数据搬移/缓冲管理与引擎计算分开，不复制 lk_moe 任何专有 CUDA。

### 更简单备选（若解码不需要 CUDA-graph）
纯 CPU 部署时让 fork 的 CPU 层**不进 CUDA-graph capture**（对离图层关 cudagraph，或让它走非 capture 分支）→ decode 直接走 `_cpu_prefill` 型路径，连补丁都不用打。**在最终集成测试会话内实测权衡。**

## 本模型的服务引擎（关键，已锁定）
`DeepSeek-V4-Flash-0731` `expert_dtype=fp4` + `moe_quant_algo=None` → fork 路由到 **`Mxfp4MoEMethod` → `_process_mxfp4` → `lk_moe.MOE_MXFP4`**（不是 NVFP4）。
喂送：w13/w2 打包 uint8 [E][2I][H/2]、[E][H][I/2]，**scale 为原始 fp8_e8m0 一字节指数** [E][2I][H/32]/[E][H][I/32]（`2^(byte−127)`），groupN=1、**groupK=32**，无 global scale。
→ 替换对应类：`MOE_MXFP4`（xiaotu-moe 已实现 e8m0 解码，真实 A/B max rel 3.4e-4）。

## 落地步骤（最终集成会话执行）
0. 预备：确保 fork 的 Python `import xiaotu_moe` 可用（`PYTHONPATH=<xiaotu-moe 仓库根>` 或 `pip install -e .`）。**停掉当前 vLLM 服务后再改代码/重启。**
1. `FORK_DIR=<路径> integration/apply_integration.sh` 一次应用两块补丁：
   - `routed_experts.swap_xiaotu.patch` —— 把所有 `lk_moe.MOE*`/`lk_moe.MOEConfigV2` 的**模块级构造**换成 `xiaotu_moe.*`（`import` 也换成 `import xiaotu_moe`）；`self.lk_moe` 实例属性及其 `cpu_prefill/cpu_decode/gpu_prefill` 方法调用**保持原样**（因为 `self.lk_moe` 现在是 xiaotu-moe 实例）。
   - `routed_experts.cpu_decode.patch` —— `_cpu_decode` 桥接（下节）。
   - `git diff` 审阅确认后再启动。类名映射自动成立（`MOE_MXFP4`→`MOE_MXFP4`、`MOE_NVFP4`→`MOE_NVFP4`、`MOE_WNA16`→`MOE_WNA16`、`MOE_BF16`→`MOE_BF16`、`MOE_FP8`→`MOE_FP8`，FP16 后缀同理）。
2. 启动服务（仅 GPU2：`CUDA_VISIBLE_DEVICES=2`）。全量 43 层数值回归：同一 prompt/batch，闭源 lk_moe vs xiaotu_moe 输出逐层对比（阈值按量化路径：BF16 <1e-3、FP4/MXFP4 <1e-2 量级）。
3. 性能回归：同吞吐/首 token 延迟对比（CPU MoE 带宽 regime，基准见计划文档）。
4. `check_nan_in_output` 已由 fork 侧处理，无需引擎内做。

> 若要临时切回 lk_moe：`git checkout -- vllm/model_executor/layers/fused_moe/routed_experts.py` 即可（补丁都不动其它文件）。

## 验证基线（本 repo 已完成的引擎级验证）
| 验证 | 结果 |
|---|---|
| 5 变体 × cp310/cp312 数值 verify | ALL PASS |
| 真实 layer-1 `MOE_NVFP4`(fp32+global) vs golden | max rel 8.7e-6 |
| 真实 layer-1 `MOE_MXFP4`(raw e8m0, fork 喂法) vs golden | max rel 3.4e-4 |
| 真实 lk_moe `MOE_MXFP4`(raw e8m0) vs golden | 中位 1.0001，尾部 ±22%（lk_moe 怪癖，xiaotu-moe 更准） |
| 真实 lk_moe 喂 fp32 scale | 3e4 爆炸 → 证 lk_moe 按字节读 scale |

## 风险
- `_cpu_decode` 桥接在 capture 内做两次 GPU↔CPU 拷贝 + `synchronize`，会拖慢 decode 吞吐（这是把 GPU 指针改为纯 CPU 计算的根本代价，与闭源 lk_moe 的 GPU 辅助拷贝同量级）。
- 若用户实际部署是 GPU-resident（`LVLLM_GPU_RESIDENT_MOE_LAYERS` 设了层号），则 MoE 层走 vLLM 标准 GPU MoE，CPU xiaotu-moe 完全不参与 —— 此时集成只验证 GPU 路径与 xiaotu-moe 的替换本身不出错即可。

## 相关脚本
- `apply_integration.sh`（一键两补丁）、`routed_experts.swap_xiaotu.patch`、`routed_experts.cpu_decode.patch`（本目录）
- 引擎级 A/B：`scripts/ab_test_real_mxfp4.py`、`scripts/ab_test_real_nvfp4.py`、`scripts/_ab_real_lkmoe_mxfp4.py`
- 性能基准：`scripts/bench_mxfp4.py`
