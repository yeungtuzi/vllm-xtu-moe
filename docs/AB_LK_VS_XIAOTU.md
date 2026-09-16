# A/B: lk-moe(LvLLM 参考实现) vs xiaotu-moe(本插件)—— §483

日期:2026-09-16 · 机器:2× EPYC 9654(192 物理核,SMT off,**2 NUMA node / NPS1**)+ 3× A100-PCIE-40GB
模型:`DeepSeek-V4-Flash-0731`(ModelScope 快照) · TP=2(GPU 0,1) · 客户端:`vllm bench serve`
(`bench_lat.sh`,512-in/128-out,N=8,**token 级 random 数据集**,不依赖 chat template)

---

## 1. 设计与为什么这样设计

| 维度 | 取值 | 理由 |
|---|---|---|
| 环境 | **两个 arm 都在 conda env `lvllm`(lvllm-2.5 = vLLM `71888f507a` + lk_moe 2.4.2)** | 基座对两边完全相同 ⇒ **唯一变量 = CPU MoE 引擎**。若一边用 lvllm、另一边用我们自己的 env,差异无法归因到引擎 |
| 模型 / GPU / TP / 请求协议 | 完全相同 | — |
| CPU 线程 | **两边都是 60** | 参考的规则是"物理核 ÷ GPU 数"= 96;我们的规则是"每 CCD 4-5 核"⇒ TP=2 每 rank 12 CCD × 5 = 60。**取同一值**,否则线程数本身就能造成两倍差异 |
| 常驻层 | 两边都不设 | 参考发布的启动脚本里没有 GPU 常驻层,故公平起点 = 都没有 |
| 投机解码 | 默认关(`SPEC=1` 可开) | 参考发布参数里有 dspark,但它会引入"投机解码"这个额外变量;引擎对比先要 plain |
| 我们的插件如何进 lvllm 环境 | `site-packages/zz_xiaotu_plugin.pth`,**受 `XTU_PLUGIN=1` 门控** | 不需要 `pip install`;arm A 不设该变量 ⇒ 插件根本不加载。自检:门控关时不加载、开时必须加载 |

启动脚本:`scripts/ab_lvllm_vs_xiaotu.sh {a|b|both|report}`;产物 `report/tuning/raw/ab_lvllm_{a,b}_*`,
日志 `report/tuning/logs/abl_{a,b}.log`。

**两条护栏(都被真实踩到过一次,现在写死在脚本里)**:
1. **纯度断言**:arm A 的日志里若出现 `vllm-xtu-moe` ⇒ 立刻判本次 A/B **作废**;
   arm B 必须出现。第一次跑 arm A 时它就被污染了,根因见 §499
   (`python -m` 把 CWD 放进 `sys.path`,仓库根的 `vllm_xiaotu_moe.egg-info/` 被
   `importlib.metadata` 当成已安装发行版 ⇒ `load_general_plugins()` 自动加载我们的插件)。
   **修复:启动前 `cd /tmp`**(我们的 `serve_mainline.sh` 本来就有这一步)。
2. **停滞检测**:日志 `STALL_S`(默认 420s)不动就打印 GPU util / CPU top / 日志尾部并判失败
   —— 否则一个 GPU 侧死锁会白等满 40 分钟超时。

---

## 2. 结果(性能)

参考配置 = `LVLLM_RELEASE_NOTES` + `commands/dsv4_0731_serve_tp2_3090_dspark.sh` 去掉投机解码,
并按本机做了两处最小调整:线程 96→60(见上)、`--kernel-config enable_jit_warmup=false`
(本机是 SM80,与我们的脚本同因)。

| arm | 环境 | 配置 | C=1 聚合 tok/s | C=1 TPOT | C=1 TTFT | C=4 聚合 tok/s | 完成 |
|---|---|---|---|---|---|---|---|
| **A: lk-moe(参考)** | `lvllm` | 参考配置:`gpu_util .90` / `maxlen 32768` / `MBT 4096` / `VLLM_COMPILE`+`FULL_DECODE_ONLY` / 无常驻层 / 60 线程 | **21.87** | **26.43 ms** | **2495 ms** | **33.95** | 8/8 |
| **B: xiaotu-moe** | `lvllm` | 与 A **逐字相同** | ✗ 起不来(见 §3) | — | — | — | — |
| B′: xiaotu-moe(不是同配置,仅供数量级) | `vllm-xiaotu-moe` | 我们自己的 `serve_mainline.sh`:`gpu_util .80` / `maxlen 8192` / `MBT 256` / 12 层 GPU 常驻 / 60 线程 | 13.03 | 49.40 ms | 3551 ms | 29.71 | 8/8 |

**怎么读这张表**:
* A vs B′ **不是**同配置,不能直接下"谁快谁慢"的结论 —— B′ 的 `MBT=256`、没有
  `VLLM_COMPILE`/`FULL_DECODE_ONLY` 都是**对解码不利**的设置(参考的优化建议就是把
  `compilation-config` 设成 `{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_DECODE_ONLY"}`),
  而 B′ 多出的 12 层 GPU 常驻对解码有利。两者方向相反,净效应未知。
* **唯一可以直接说的**:在本机本模型上,**参考实现在它自己的配置下 C=1 是 21.87 tok/s
  (TPOT 26.4 ms)**;我们的引擎目前**没有**在参考配置下跑起来(原因见 §3),
  所以"同配置 A/B"这个问题**尚未有答案**。
* 参考的 C=1 21.9 tok/s 与它 release notes 里 V4-Flash 在 2×3090 上的 30.6-31.1 t/s 同量级
  (本机是 A100-40G、2 NUMA node、LK_THREADS 60 而非 48,不是同一硬件口径)。

---

## 3. arm B 为什么起不来:一条**逐步逼近**的链,每一步都有证据

### 3.0 第一步(已修):专家权重被建在 **GPU** 上 ⇒ CUDA OOM
```
File ".../vllm_xiaotu_moe/mainline_shims.py", line 541, in create_weights
File ".../vllm/model_executor/layers/quantization/mxfp4.py", line ..., in <lambda>
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB.
GPU 0 has a total capacity of 39.49 GiB of which 651.50 MiB is free.
```
**根因(lvllm-2.5 的 `mxfp4.Mxfp4MoEMethod.create_weights`,逐字)**:
```python
device = torch.cuda.current_device() if current_platform.is_cuda_alike() else "cpu"
new_tensor = torch.zeros
if isinstance(layer, RoutedExperts) and not layer.is_gpu_resident_layer:
    device = "cpu"          # ← 唯一能让专家权重留在 CPU 的分支
    new_tensor = torch.empty
...
new_tensor(..., device=device)      # ← 显式 device= ⇒ 我们 `with torch.device("cpu")` 无效
```
而 `is_lk_moe_gpu_resident_layer()` 在 `LVLLM_MOE_NUMA_ENABLED=0` 时**恒返回 True**
(`vllm/envs.py:2551: if not is_lk_moe_feature_enabled(): return True`)。
我们的混合模式**必须**关掉 lk_moe ⇒ 条件永远不成立 ⇒ 1 GiB 的专家张量往 CUDA 上建
(43 层 × 1.59 GiB/rank 根本放不下,`gpu_util` 降到 0.80 也没用)。

**已修**(`mainline_shims._patch_quant_method_cls.create_weights`):在调用原实现前
`layer.is_gpu_resident_layer = False`(**只在属性存在时改** ⇒ 主线没有该属性,行为逐字不变)。
修完**OOM 消失**,直接推进到 3.1 —— 这一步是可验证的。

### 3.1 第二步(**未修**,是真正的移植点):CPU MoE 执行被**硬绑在 lk_moe** 上
```
File ".../vllm/model_executor/layers/fused_moe/runner/moe_runner.py", line 664, in _apply_quant_method
File ".../vllm/model_executor/layers/fused_moe/routed_experts.py", line 1841, in _cpu_prefill
    self.lk_moe.cpu_prefill(...)
AttributeError: 'NoneType' object has no attribute 'cpu_prefill'
```
lvllm-2.5 的 `RoutedExperts._cpu_prefill`(`routed_experts.py:1831-1841`)直接调
`self.lk_moe.cpu_prefill(...)`;`lk_moe` 关闭时它是 `None`。
**这里没有"换后端类"的缝** —— 与主线不同(主线把 CPU 专家做成
`FusedMoEExpertsModular` 后端类,我们的 `register_mixed_cpu_backend()` 换掉那 4 个类就接管了)。
⇒ 想在 lvllm 里跑我们的引擎,**必须**加一个适配层,二选一:
* (a) 让插件 patch `RoutedExperts._cpu_prefill`(以及对应的 GPU-prefill 分支)去调我们的引擎;
* (b) 给 `self.lk_moe` 绑一个**鸭子类型的替身**:暴露 lk_moe 的 `cpu_prefill`/`cpu_decode`/`gpu_prefill`
  签名。
  **好消息**:(b) 的成本比看起来小 —— 我们**已经有** `xiaotu_moe/gpu_prefill_bridge.py`
  给引擎类包装了 lk_moe 签名的 `gpu_prefill(x_ptr, out_ptr, ids_ptr, wts_ptr, qlen, k, stream)`
  (当初就是为 lk 编排写的)。缺的只是 `cpu_prefill` / `cpu_decode` 这两个同风格的入口。

### 3.2 Mode A(OOT 覆盖模型类)+ 参考的 `VLLM_COMPILE` + `MBT=4096` ⇒ **GPU 侧死锁**
* 现象:86 个引擎全部建好,随后卡在 profile run;日志最后一行
  `08:14 … DSA indexer decode path`(之后只剩 `shm_broadcast` 的 60s 告警)。
* 现场证据(不是猜):
  * 两个 worker **瞬时 CPU = 0**;`gdb -p` 主线程栈全在 **`libcuda.so` 里 `sched_yield`**;
  * `nvidia-smi`:两个 GPU **util 0%**、显存停在 **7.4 GiB**(只有权重,**没有 KV cache**);
  * 没有 `WATCHDOG fired`(故**不是**我们引擎线程池挂住 —— 那种会 300s 后 dump+abort)。
  ⇒ 定性:**GPU 侧等不到对端**(TP=2 集合通信/图捕获层面的死锁),不是 CPU 引擎在算。
* 最可能的原因:Mode A 此前只在主线的 `CompilationMode.NONE` 下验证过,
  **`VLLM_COMPILE` 这条路径从未验证**(我们的 `serve_mainline.sh` 用的就是 `NONE`)。
  **应当先修成"检测到该组合就显式拒绝启动",而不是死锁** —— 死锁是最坏的失败形态。

### 3.3 结论(关于"同配置 A/B")
**在 lvllm-2.5 上跑通我们的引擎需要一次真正的移植**(3.1 的适配层;3.2 的编译模式支持),
**不是调参**。3.0 是这条路上第一个已修掉的障碍。
在此之前,**同配置性能对比无法给出**;把 B′ 的数字当"同配置结果"是不诚实的。

---

## 4. 结果(正确性)

* **arm A(参考,lvllm-2.5 + lk_moe)**:5 条固定 greedy prompt 全部给出正确答案
  (`report/tuning/raw/ab_lvllm_a_greedy.json`):
  `cap_fr → 'Paris'`、`math → '17*23 = 391 / 391 + 19 = 410'`、
  `code → "sum(...)" one-liner`、`zh → 正确的 MoE 一句话解释`、
  `list → '2, 3, 5, 7, 11, 13, 17, 19'`。服务端 greedy 无异常。
* **arm B**:未起服务,故**没有**同配置的正确性对比。
* 严格的**逐 token 相等**对比这次也做不了:arm A 的服务带了
  `--default-chat-template-kwargs '{"enable_thinking": false}'`(参考脚本里有),
  而我们的 V4 基线服务没带 ⇒ 两边的 prompt 渲染不同(我们会输出 `thinking` 文本),
  逐 token 比较无意义。**要做得先把 chat template kwargs 对齐**,这是下一步的第一个动作。
* 我们引擎侧**已有的**正确性证据(与本次 A/B 独立):
  * 内核数值门禁 `test_block23_equiv.py` = **`OK=7 BAD=1`**(`me=1 max_rel 1.873e-02`,
    与 v0.1.0 起的基线**逐字相同**);
  * 引擎逐位确定性 `test_engine_determinism.py 12` = **11/11 bit-identical**;
  * V4 服务端 greedy 5 连测 = **4/5 逐字节相同**(已知的 TP=2/EP 归约非确定性)。

---

## 5. 下一步(按优先级)

1. **对齐 chat template kwargs 后重跑两边的 greedy 探针** ⇒ 拿到真正的正确性对比
   (这一步不需要 arm B 在 lvllm 里跑起来:可以先用"我们的 env + 我们的脚本 + 同一个 kwarg").
2. 移植 Mode B 的 device-loading 覆盖到 lvllm-2.5 的 `mxfp4/fp8/int4.create_weights`
   ⇒ 让 arm B 至少在 `gpu_util` 0.80 下起得来(这是同配置 A/B 的前置条件)。
3. Mode A 适配 `VLLM_COMPILE`(或明确记录"我们的 Mode A 不支持 VLLM_COMPILE",
   并让插件在检测到该模式时**显式报错**而不是死锁 —— 死锁是最坏的失败形态)。
4. 重新做 arm A:用与成功配置相同的 `MBT/maxlen/gpu_util`,把 A/B 的**唯一变量**真正压到引擎上。
