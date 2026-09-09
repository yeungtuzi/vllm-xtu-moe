# 夜间工作交接(2026-09-08 夜 → 09-09 晨)

## 一、今晚达成(已验证)

### 1. 通用 GPU/CPU Mixed Mode —— 真实模型 E2E 跑通 ✅
- 模型:gpt-oss-20b(GptOssForCausalLM,24 层,32 专家,topk4,hidden2880,**mxfp4**,A100 可跑)。
- 命令:`VLLM_EXPERTS_LOAD_DEVICE=cpu`,GPU2,--load-format auto。
- 结果:加载 8.26s → `Using XiaotuCPUExperts` → 构造 88.7s → **生成 31.7 tok/s**。
- 链路实证:model → MoERunner → Mxfp4MoEMethod.apply_monolithic → moe_kernel → **XiaotuCPUExperts.apply → engine.cpu_decode** → out。
- 意义:**注意力在 GPU、MoE 在 CPU(xiaotu AVX512-VNNI,无 AMX)首次真实端到端跑通**。

### 2. 主线改动(可 PR)
- `vllm/envs.py`:`VLLM_EXPERTS_LOAD_DEVICE` + 5 个 `VLLM_TRITON_MLA_SPARSE*`。
- `routed_experts.py`:混合模式下专家参数构造落 CPU(`with torch.device("cpu")`)。
- `oracle/mxfp4.py`:①混合模式**强制优先 CPU 后端**;②CPU 分支跳过 AMX prepack(返回原始权重)。
- 插件 `vllm_xiaotu_moe/mixed_experts.py`:`XiaotuCPUExperts(CPUExpertsMxfp4)` + `register_mixed_cpu_backend()`。

### 3. 修的关键 bug(4 个)
1. oracle 未优先 CPU → A100 选 Marlin → "b_q_weight is not on GPU";
2. CPU 分支仍做 AMX prepack(`convert_weight_packed` 不存在);
3. `w1_scale` 是只读 property,不可赋值;
4. **cpu_decode 必须传 `data_ptr()` 整数**(binding 的 `as_ptr` 对 torch 张量返回 nullptr →
   `cudaMemcpyAsync(nullptr)` → invalid argument)。已用 `scripts/cpu_decode_test.py` 复现+验证。

### 4. SM80 移植:素材就位(导入已验证)
- 新增并**导入通过**:`sparse_mla_env.py`(119)、`sparse_mla_kernels.py`(3517,补了
  `fp8_utils._e4m3_uint8_to_f32`)、`ops/{sm12x_deep_gemm_fallbacks,sm12x_mqa,fp8_einsum}.py`。
- `is_ampere_or_ada()` 在 A100 返回 True。
- 清单见 `docs/sm80_port_inventory.md`。**待办 = wiring**(flashmla.py 等 5 个已分化文件)。

## 二、待办(按优先级)

1. **【高】混合模式数值正确性**:gpt-oss CPU 输出乱码,GPU(Marlin)输出正确 " Paris."。
   **根因已确认**:`models/gpt_oss.py:389` 是 `has_bias=True, activation="swigluoai"`,
   而引擎实现的是**标准 silu + 无 bias** → 不匹配。**不是引擎 MXFP4 数学错**(M1 已用 DS-V4
   真实权重对 torch 参考,rel err<2e-3)。
   **关键推论**:DS-V4/GLM 用的正是**标准 silu + 无 bias**,与引擎假设一致 → 对真实目标模型,
   MoE 数值无引擎侧障碍;唯一障碍是它们的 **sparse-MLA 注意力在 A100 无后端**(见第 3 项)。
   => 数值验证应换 silu+无 bias 的模型,或等 SM80 注意力打通后直接用 DS-V4/GLM。
2. **【高】吞吐实测(带批)**:单流 31.7 tok/s;需测 B≥64 是否达 >100 tok/s(参考 E=256 实测 110-167)。
   静默原则:性能测量前需停对 prod 8070 的调用(用户负责)。
3. **【中】SM80 wiring**:把 Triton sparse-MLA 接入主线 flashmla.py 等,让 DS-V4/GLM 在 A100 跑。
4. **【低】GLM 权重补齐**:hf-mirror 下载 19 分片 + tokenizer(scripts/glm_fill_dl.out)。

## 三、环境/资源
- 补装:torchvision 0.28.0+cu130、flashinfer-python 0.6.18.post1。
- gpt-oss-20b:`/home/user/.cache/hf-models/gpt-oss-20b`(13.76GB,完整)。
- 脚本:`scripts/{model_mixed_test.py,cpu_decode_test.py,mixed_load_device_test.py}`。
- 关键日志:`scripts/{gptoss_cpu_real.out,gptoss_gpu_real.out,gptoss_real.out}`。
- **prod 未碰**(8070/GPU0+1);全程 GPU2/8071。
