# vllm-xtu-moe v0.2.2 — 支持 GLM-5.3-Flash

**发布日期**:2026-09-19
**主题**:在 2×A100-40GB(SM80,TP=2)上把 **GLM-5.3-Flash(321B / 18B active,原生 FP8 block-128)**
做成可交付的端到端服务,并把长 prompt 的预填充从 CPU 挪到 GPU。

---

## 1. 一句话

`zai-org/GLM-5.3-Flash` 在 2×A100-40GB + TP=2 上跑通并交付:
**上下文 256K × 2 路并发**,4K prompt 的 **TTFT 29.3 s → 22.8 s(1.29×)**,
解码不变(≈22 tok/s),长文检索与并发正确性均已实测。

## 2. 新增能力

### 2.1 FP8 GPU 预填充(核心)

专家权重常驻主机(283.5 GiB,单份 NUMA 分片),长 prompt 时**逐层把该层专家权重流式搬到
GPU** 上算。新增 `vllm_xiaotu_moe/gpu_prefill_fp8.py`:

* Triton 内核:`gate_up` / `down`,e4m3 **1 字节/元素** + **block-128 fp32 scale**
  (与 MXFP4 的 nibble + e8m0 block-32 完全不同);A100 无 FP8 张量核 ⇒ 解码到 bf16 再 `tl.dot`;
* 权重装配直接从引擎自有的 NUMA 紧凑分片重建(**不需要第二份专家副本、不多占主机内存**),
  K-major 目标按 `cudaMemcpy2DAsync` 直接落位;
* 后端按**引擎类别**自动选择(`MOE_MXFP4 → gpu_prefill`,`MOE_FP8 → gpu_prefill_fp8`),
  两个模块同名同签名入口,上层流式分支不再关心量化格式;
* **装配与 attention 重叠**(side stream + 一个 event),这是本轮真正的性能来源。

| 每层(GLM 形状,TP=2) | 实测 |
|---|---|
| 权重装配(纯 H2D 3.62 GB/rank) | **207 ms**(已到本机 26.86 GB/s 的 H2D 天花板) |
| GPU MoE 内核 | **23 ms**(比 CPU 引擎快约 10×) |
| 与 attention 重叠的收益 | **1.20×**(同构建 A/B) |

> 装配**不是**"搬得不够快":实测 1-D 连续与 pitched 2-D `cudaMemcpy2DAsync` **同速**
> (26.86 GB/s),`cudaHostRegister` 锁页、`numactl` 交错、两个 rank 并发都**不改变**它。
> 所以优化点只能是**别让搬运排在 attention 后面**。

### 2.2 服务配置:256K 上下文 × 2 路并发

`scripts/serve_glm53_mainline.sh` 默认值更新为
`MAXLEN=262144 / MBT=8192 / SEQS=2`,GPU 预填充阈值 `GPU_PREFILL_MIN=1500`。

| 项 | 实测 |
|---|---|
| Available KV | 11.55 GiB |
| **GPU KV cache size** | **988,081 token** |
| 256K 下的 KV 并发 | **3.77×**(两路 256K 只占 53%) |
| 3 路 ~8K 并发(限两路) | 2 个 35.9 s + 1 个 47.0 s,**三个密钥全部正确** |

**服务级 benchmark(官方 `vllm bench serve`,随机数据 + `--ignore-eos`,每格不同 seed)**:

| 并发 | prompt / output | out tok/s(含 TTFT) | TTFT 均值 | TPOT 均值 | 完成 |
|---|---|---|---|---|---|
| C=1 | 256 / 128 | **16.18** | 2022 ms | **46.37 ms** | 8/8 |
| C=2 | 256 / 128 | 19.36 | 3287 ms | 78.13 ms | 8/8 |
| **C=1** | **4096 / 64** | 2.49 | **22764 ms** | 46.95 ms | 2/2 |
| C=2 | 4096 / 64 | 2.59 | 34162 ms | 243.21 ms | 4/4 |

对照 v0.2.1 的配置(4096 上下文 / MBT=1024 / 阈值 4096 ⇒ **全 CPU 预填充**):

| 口径 | v0.2.1 | **v0.2.2** |
|---|---|---|
| 4096-in / 64-out C=1 TTFT | 29320 ms | **22764 ms(1.29×)** |
| 256-in / 128-out C=1 out tok/s | 16.20 | 16.18(**解码不变**) |
| 256-in / 128-out C=1 TPOT | 45.03 ms | 46.37 ms |
| 上下文 × 并发 | 4096 × 4 | **262144 × 2** |

> `4096/64` 的 C=2 那格 TPOT 243 ms 是 chunked prefill 的正常代价(长 prompt 的 chunk 与
> decode 步交错);长 prompt 场景建议「一路长 + 一路短」。

上下文阶梯(util 0.90、bf16 KV):**256K ✅ / 512K ✅(843,055)/ 704K ✅(763,177,仅 1.06× 并发)/
768K ❌ / 1M ❌**。上限 ≈ **733K**;1M 需要 **fp8 KV**(未实现,见 §4)。
注意 GLM 的 KV 单价是 **~11.9-12.3 KB/token**(只有 11 层 NoPE 稀疏 MLA 带 per-token KV),
**不是** DeepSeek-V4.1 的 ~40 KB/token。

### 2.3 引擎缺陷修复:e4m3 次正规数解码(重要)

`kernels/fp8_dequant.hpp` 里两个**免查表算术解码器**(AVX-512 `e4m3x16_to_fp32`、
AVX2 `e4m3x8_to_fp32_arith`)把次正规数写成 `2^-9·(1+m/8)`,正确值是 **`m·2^-9`**:

| 字节 | 正确 | 修复前 |
|---|---|---|
| `0x00` | 0 | **`2^-9`**(每个"零权重"都带直流偏置) |
| `0x07` | `7·2^-9` | `2^-9·1.875`(**小 3.7×**) |

标量/LUT 路径一直是对的,所以问题藏在只有 FP8 内核走的那两条 SIMD 路径里。
真实检查点上影响很小(≈1.3e-5 每权重、≈1e-4 每层输出,block-128 的 `amax/448` 把它压住了),
所以历史上所有精度门禁都没抓到;它是在 FP8 GPU 路径 vs CPU 引擎对拍(3.4e-2)时暴露的。

* 修复:8 项 `m*2^-9` 表 + `vpermps`(每 16 个权重多 1 uop,FP8 M-curve 在噪声内不变);
* **新增门禁** `scripts/test_fp8_decode_conformance.py`:256 个 e4m3 码字 ×
  {avx512, avx2, scalar} 对照 `torch.float8_e4m3fn` —— 修复前每 ISA **16 处不符**,修复后 **0**;
* 旁证:CPU 引擎 vs 精确数学 **3.4e-2 → 1.5e-6**,恒等探测**逐位相同**。

### 2.4 可选:AVX512-BF16 FP8 内层(默认关闭)

`XIAOTU_MOE_FP8_BF16_MMA=1` 让 FP8 内层改走 `vdpbf16ps`(32 个 bf16/指令、激活无需转换):

| M | 1 | 6 | 12 | 28 | 57 | 114 |
|---|---|---|---|---|---|---|
| 加速 | 0.87× | 1.17× | 1.17× | 1.19× | 1.19× | **1.20×** |

M≥6 稳定快 17-20%,但 M=1 反而慢 13% ⇒ 分发时用 `M > 4` 闸门,**单流解码仍走精确 fp32 路**。
全 CPU 预填充端到端 **26.60 → 23.72 s(1.12×)**。
**代价**:权重必须舍入到 bf16(fp32 路 vs bf16 路 rms_rel 3.63e-3),**因此默认关闭**。

### 2.5 可选:ping/pong 跨层预取(默认关闭,**不推荐**)

`XIAOTU_GP_ASM_PREFETCH=1` 把下一层的装配提前到本层计算期间。

* 收益只有 **1.07×**(22.66 → 21.10 s):免费的同层重叠已经把装配藏进 attention,
  跨层预取额外只省得下与 MoE 内核重叠的那部分;
* 代价 **+3.38 GiB/rank**,需要把 `--gpu-memory-utilization` 从 0.90 降到 0.82;
  **在 util 0.88 下实测会把引擎 OOM 打死**(38.02/39.49 GiB 已分配后一个 30 MiB 请求失败);
  现在分配前会检查"槽位 + 4 GiB 工作区",不满足则拒绝并退回免费的那份重叠;
* **PCIe 5.0 及以上不建议开**:装配时间减半后会低于身后的计算,收益趋近 0,显存却照付。

## 3. 验证与门禁

| 门禁 | 结果 |
|---|---|
| `scripts/test_fp8_decode_conformance.py`(新) | 256 码字 × 3 条 ISA 路径,**0 不符** |
| `scripts/test_gpu_prefill_fp8_assembly.py`(新) | KP-major 装配**逐字节相同**(并当场抓出过一个 w13 源行距错误) |
| `scripts/test_gpu_prefill_fp8_vs_cpu.py`(新) | GPU 预填充 vs CPU 引擎 **`[OK]` rms_rel 4.43e-3** |
| `scripts/test_fp8_bf16mma_equiv.py`(新) | fp32 路逐位可复现;两路 rms_rel 3.63e-3 < 1e-2 |
| `test_block23_equiv.py`(MXFP4/V4 路径) | **OK=7 BAD=1**(与基线数字完全一致,未被触碰) |
| `test_engine_determinism.py` | **11/11** 逐位一致 |
| `check_engine_aligned.sh` | 数值门禁通过;DEDUP=12 0.50 ms/层 0.78×、DEDUP=23 0.62 ms/层 0.90×(PASS) |
| 真实服务 | 4K prompt TTFT 22.8 s;29,746-token 长文密钥**检索完全命中**;3 路 ~8K 并发(限两路)密钥全对;**0 OOM** |

## 4. 已知限制

* **1M 上下文不可达**(bf16 KV):768K 起不来(vLLM 自报上限 733,312),1M 需要 11.57 GiB 的
  注意力 KV,而该 maxlen 下只有 6.87 GiB。唯一出路是 **fp8 KV**(SM8x 稀疏 MLA 的 fp8 变体 +
  NoPE 感知的 528 B blob;现成的 `fp8_ds_mla` 656 B blob 只到 ~948K)。
  **决定:本硬件上以 256K × 2 路交付,不做 fp8 KV、不追 1M**;512K/704K 的「能起」只是
  能力记录(704K 仅 1.06× 并发),不作为目标。
* `--mamba-ssm-cache-dtype` 对容量**无**帮助;TP>2 对 MLA 的 KV **无**帮助(每 rank 复制),
  TP=3 不整除 64 头,SM8x 不支持 DCP。
* 同一 prompt 重复请求的贪心输出**会**偶发 token 级抖动(near-tie 翻转,与 batching /
  prefix-cache 命中有关)。引擎层确定性 11/11,预取开/关输出逐字节相同;
  该抖动是服务栈既有现象,不作为验收项。
* 交付配置在 util 0.90 且 GPU 预填充缓冲就位后,显存只剩 ~1.26 GiB 工作区。实测够用
  (长 prompt 被 KDA 切成 ≤2176 token 的 chunk),预检不过时会优雅退回 CPU(慢约 1.4×)。

## 5. 兼容性

* **主线 vLLM**:本轮未改主线树;插件侧新增文件/脚本,`MOE_MXFP4`(DeepSeek-V4/V4.1)路径
  只有"按引擎类别选后端"这一处间接化改动,数值门禁逐字复现基线。
* **主机内存**:FP8 GPU 路径复用引擎自有分片,**没有**第二份专家副本。
* **显存**:GPU 侧缓冲为显存开销(装配缓冲 ~6.75 GiB,首层长 prefill 时惰性分配)。

## 6. 安装

**v0.2.2 起首次附带 whl**(此前的版本只有源码):

```
vllm_xtu_moe-0.2.2-cp312-cp312-manylinux_2_34_x86_64.whl        # 4.65 MB
```

> 校验和见 Release 页面的 asset 摘要。**wheel 构建不是逐字节可复现的**(zip 里带时间戳),
> 所以这里不固定 sha256 —— 换了构建机/时间就会变,但内容(6 个变体 + 包代码)一致。

```bash
pip install ./vllm_xtu_moe-0.2.2-cp312-cp312-manylinux_2_34_x86_64.whl   # 需先装好 vLLM
```

* wheel **自带 6 个 ISA 变体**(scalar / avx2 / avx512_base / avx512_vnni / avx512_bf16 /
  avx512_bf16_vbmi,g++-16 构建),导入时按 CPU 自动选最高可用 —— **不需要本地编译器,
  也不再需要跑 `build_engine_variants.sh`**;
* 约束:Python **3.12** + Linux **x86-64**(glibc ≥ 2.34);`.so` 依赖 `libcudart.so.12` 与
  驱动(装了 vLLM/torch 的环境天然满足);vLLM 插件入口
  `vllm.general_plugins → vllm_xiaotu_moe.hybrid_model:register` 已在 wheel 里注册;
* 需要自定义内核开关或换编译器时,仍走源码路径(见 `docs/RUNBOOK.md` §2 方式 B)。

## 7. 复现

```bash
CXX=g++-16 PYTHON=$(which python) bash scripts/build_engine_variants.sh && pip install -e .
bash scripts/serve_glm53_mainline.sh          # 默认即 256K × 2 路 + GPU 预填充
python scripts/probe_ttft.py                  # PORT=8073 LENS=4300
/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/vllm bench serve \
  --backend openai --host 127.0.0.1 --port 8073 --model GLM-5.3-Flash \
  --dataset-name random --random-input-len 4096 --random-output-len 64 --ignore-eos \
  --num-prompts 2 --max-concurrency 1 --save-result --result-dir /tmp
```

详细数据:`docs/BENCHMARKS.md` §4.1、`docs/MODEL_GUIDES.md` §2;内部全过程记录:
`dev-docs/GLM53_SM80_PLAN.md` §16-23(该目录不随仓库发布)。
