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

## 2. 结果(性能)—— **同 env、同参数、逐字对齐的 A/B 已经拿到**

参考配置 = `LVLLM_RELEASE_NOTES` + `commands/dsv4_0731_serve_tp2_3090_dspark.sh` 去掉投机解码,
并按本机做了两处最小调整:线程 96→60(见上)、`--kernel-config enable_jit_warmup=false`
(本机是 SM80,与我们的脚本同因)。

| arm | C | 聚合 tok/s | TPOT | TTFT | 完成 | B/A |
|---|---|---|---|---|---|---|
| **A: lk-moe(参考)** | 1 | **21.87** | **26.43 ms** | **2495 ms** | 8/8 | — |
| **B: xiaotu-moe** | 1 | 10.37 | 61.06 ms | 4586 ms | 8/8 | **0.47×** |
| **A: lk-moe** | 4 | **33.95** | **49.15 ms** | 7075 ms | 8/8 | — |
| **B: xiaotu-moe** | 4 | 16.22 | 100.38 ms | 14940 ms | 8/8 | **0.48×** |

两边**逐字相同**的东西:`lvllm` 环境 / 同一个模型快照 / TP=2(GPU 0,1)/ `gpu_util .90` /
`maxlen 32768` / `MBT 4096` / `--kv-cache-dtype fp8_ds_mla` /
`--compilation-config {mode: VLLM_COMPILE, cudagraph_mode: FULL_DECODE_ONLY}` /
`--enable-prefix-caching --enable-chunked-prefill` / THREADS=60(`LK_THREADS` 与
`XIAOTU_MOE_THREADS` 取同一值)/ **两边都不设 GPU 常驻层** / 同一个客户端与协议。

### 2.1 结论(可以说出口的)
1. **纯解码一步的代价:lk_moe 26.43 ms,我们 61.06 ms ⇒ 我们慢 ~2.31×**(C=4 时
   49.15 vs 100.38 ms,同样 ~2.04×)。这是引擎自己的活儿,**不受 GPU 预填充开关影响**,
   所以是本轮最干净的一个数字。
2. **预填充 TTFT:2495 vs 4586 ms(~1.84×)** —— 但这里有**已知的不对称**:
   参考脚本设了 `LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024` + `LVLLM_GPU_PREFETCH_WINDOW=1`,
   **lk_moe 的 GPU 预填充是开着的**;而我们对应的开关(`XIAOTU_MOE_GPU_PREFILL_MIN_TOKENS`)
   **没设 = 0**,走的是 CPU 预填充。⇒ TTFT 这一项**不能**当作"同样功能下谁快"。
   (我们自己此前测过:我们的 GPU 预填充目前**打不过** CPU,所以默认关着 —— 这正是待办。)
3. 因此**"引擎算力"层面的对比用 TPOT**:**lk_moe 在本机本模型上比我们快约 2.1-2.3×**。

### 2.2 这个 2.1× 是"引擎本身"还是"引擎在这台机器上的调参"?—— 必须再走一步
两条已知线索都指向**调参/拓扑**,而不是内核:
* **§498**:同一份我们的代码,机器从 8 NUMA node(NPS4)变成 2 node(NPS1)后,
  `nshard_` 8→2,DEDUP=12 的微基准从 **0.65 → 0.95 ms/层(1.5×)**;
  而 `lk_moe` 的架构是"每个 node 一份分片",对 node 数变化可能没有那么敏感;
* 我们把两边线程数**都钉在 60**(我们的"每 CCD 4-5 核"规则)。而参考自己的规则是
  "物理核 ÷ GPU 数" = **96** ⇒ 60 对 lk_moe **可能不是它的最优点**。
  所以下一步是**线程数扫描**(见 §2.3),把"引擎质量"与"引擎调参"分开。

### 2.3 arm B 的线程数扫描(把"引擎"与"调参"分开)

`THREADS` 就是 arm B 的 `XIAOTU_MOE_THREADS`;其余参数与 §2 完全一致。

| `THREADS` | C=1 聚合 tok/s | C=1 TPOT | C=1 TTFT | C=1 完成 | C=4 完成 | 备注 |
|---|---|---|---|---|---|---|
| **60**(正式 arm B) | **10.37** | **61.06 ms** | 4586 ms | 8/8 | 8/8 | 目前最好 |
| 96 | 9.18 | 72.33 ms | 4755 ms | 8/8 | **2/8** | 更慢;C=4 有 6 个请求没完成 |
| 120 | — | — | — | — | — | **服务被自家线程池看门狗 abort**(见下) |

**结论**:
1. **我们的引擎在这台机器上的线程最优点是 ≤60**,96 更慢、120 直接不稳
   ⇒ **与 lk_moe 的 2.1-2.3× 差距不是"线程数没调对"造成的**;
2. 那么差距更可能来自**结构性原因**:`nshard_ = max(1, node/world)` 在本机(2 node、TP=2)
   等于 **1**(逐 socket 整份副本),而 8-node 时代是 4;§498 已实测同一份代码在 8→2 node 之后
   DEDUP=12 微基准退化 1.5×。**lk_moe 的分片/预取策略对 node 数变化的敏感度需要单独量**
   (它的架构是"每 node 一份分片",可能天然更适应 2 node);
3. ⚠️ **`THREADS=120` 这一次是我们引擎的又一个真 bug,必须单独修**(与"谁快谁慢"无关):
   ```
   [pool] WATCHDOG fired: gen=2086 n=192 start=454774 end=454966
                          counter=455086 remaining=1 current_gen=2086 dropped=2
     worker[0] slot=2086 …(120 个 worker)
   ```
   `counter - end = 120 = nt_` ⇒ 又一次"**所有 worker 都走完了领票循环、但有一张已领的
   区间内票没有递减**"⇒ `remaining_` 永远差 1 ⇒ 300s 后 `abort()`。**这一次 env 是干净的**
   (显式 `XIAOTU_MOE_ENV_FILE`、`SPIN_IDLE_US=0`、`THREADS` 显式)⇒ 说明 §497 里
   "陈旧 env 只是**触发器**、底层账目竞态是**真 bug**"这个判断成立。修法见 §497(5):
   flat 路径补无损账 + 发布侧 `counter_.fetch_add(n)` 原子预留 + worker 侧在 `work_mtx_`
   下复核活代后再决定丢弃。

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
