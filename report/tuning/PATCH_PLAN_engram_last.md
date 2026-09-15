# Patch 规格:Engram(ngram)最后加载 + 大权重逐层 load/slice/release

依据:**IRON_RULES R11**(用户 2026-09-15 定则)、NOTES §394。
目标主机:2×EPYC 9654 / 3×A100-40GB / 1.5 TiB(NPS=1,每 node 756 GiB)。
目标模型:DeepSeek-V4.1-Flash(748B,475 GiB,`cpu_offload=True` 的 Engram 表)。

---

## 0. 为什么必须动 vLLM,而不是只改插件

| 事实 | 证据 |
|---|---|
| Engram 的 pinned 表在**模块构造期**分配 | `vllm/models/deepseek_v41/nvidia/engram.py:266-296` `_allocate_weights()` 的两个 `torch.empty(..., pin_memory=True)`;日志 `[engram.py:255]` 出现在 `load_weights` **之前** |
| 表大小 94.42 GiB × 2 = **188.8 GiB**,pinned ⇒ 不可回收、不可换出 | `cellK` 日志 行44/46;本机无 swap |
| 专家层的处理(逐层释放)发生在**表已就位之后** | `cellK` 行53…147 | 
| 于是整个内存最紧的阶段白扛 188.8 GiB | 该阶段 `RssAnon` 一度 838 GiB,RSS 合计 1064–1116 GB |

⇒ 只靠插件 shim 无法改变"构造期分配"这个时序,必须在 vLLM 侧改动。

---

## 1. Patch A:Engram 延后到"专家权重全部处理并释放之后"

### 1.1 设计要点

Engram 的分配点在**构造期**,而 `load_weights` 在其后 —— 所以"后置"有两个必须一起解决的子问题:

1. **分配后置**:构造期不能占 188.8 GiB;
2. **加载后置**:加载器(真实权重的 safetensors 读取,或 `--load-format dummy` 的 dummy 填充)
   本来会写进构造期建好的张量。后置之后,必须在**物化之后**再补一次 Engram 的加载。

**若只做 (1) 不做 (2)**,真实权重下 Engram 表会是**未初始化内存**(bug),dummy 下也会丢掉
`dummy_weight_value` 语义。

### 1.2 建议实现

* 新增开关:`vllm_config.engram_config.load_last: bool`(默认 `False`,保持主线行为);
  或等价的环境变量。**默认关闭**是硬要求,避免影响其它模型的性能与行为。
* `ParallelEngramEmbedding.__init__` / `_allocate_weights()`:
  * `load_last=True` 且 `cpu_offload=True` 且非 `dp_shared_memory` 时,
    **只分配 meta 张量**(shape 正确、零内存),并把模块登记到一个待物化列表;
  * 其余路径**逐字不变**。
* **物化时机**:所有专家层完成 `process_weights_after_loading` **之后**。
  现成的挂载点:`vllm/model_executor/model_loader/utils.py:167`
  ```python
  if hasattr(model, "process_weights_after_loading"):
      model.process_weights_after_loading()      # ← 模型级 post-load 钩子
  ```
  在这里(或紧随其后)统一物化待定的 Engram 表。
* **物化动作**:分配真正的 pinned 张量 → 赋给 `weight` / `weight_scale_inv`
  → **重新跑一遍 Engram 的权重加载**(复用已有的 `weight_loader` / dummy 填充路径)。
* `dp_shared_memory=True` 走 `/dev/shm` 共享路径,**不在本 patch 范围**(TP=1/DP=1 下无收益)。

### 1.3 预期收益与验收

* 预期:专家阶段峰值 **−188.8 GiB**。
* 验收(必须实测):
  1. 日志里 `[engram.py:255]` 出现在**最后一条** `released ... layers.39.ffn.experts` **之后**;
  2. 专家阶段结束时的 `RssShmem` 比改动前低约 188.8 GiB;
  3. `load_last=False` 时,日志与内存曲线与改动前**逐字一致**(回归)。

---

## 2. Patch B:大权重"逐层 load / slice / release",禁止一次性全量

### 2.1 现状与缺口

* 插件侧已经做到**逐层 slice + release**(实测 40/40 层,每层放 6.59 GiB,N0TES §391/§392);
* 但**加载**仍是 vLLM 的"一次性读完全部权重"模型:**所有层**的源张量在 `load_weights`
  阶段被 materialize,逐层释放只是把峰值削平,并没有把"同时存在"的层数降下来。
* 用户要求:识别我们的标记参数,做**逐层 加载→切片→释放**;作为**单独 PR 提交**。

### 2.2 标记机制(已就位,可直接复用)

`vllm_xiaotu_moe/mainline_shims.py` 的 `create_weights` shim 会给混合模式下建在 CPU 的
**大**参数(≥64 MiB)打上属性 **`_xiaotu_cpu_expert = True`**。这就是 patch 里
"这是我们的参数"的识别依据。

### 2.3 建议实现

* 在权重加载循环(`default_loader` 的 `load_weights` 迭代器消费处)识别
  `getattr(param, "_xiaotu_cpu_expert", False)`;
* 对这些参数**不**沿用"先全部读入再统一 finalize",而是按层:
  1. 只读入该层参数 → 交给 `process_weights_after_loading`(我们的切片)
  2. 切片完成后立即释放该层源张量(插件已经会做)
  3. 进入下一层
* 需要保证:模块化链路里 `w1/w2` 被透传进 `apply()` 时**形状仍在**
  —— 这正是 `empty_strided` 修复解决的问题(NOTES §392),patch 里应沿用同一语义。

### 2.4 与 Patch A 的关系

两者**独立**、可分别提交与回归。A 解决"ngram 不该在专家阶段占内存",
B 解决"专家权重不该同时全部驻留"。合起来才是用户要的完整形态。

---

## 3. 仍未点名的问题:实测 RSS 比账面高约 600 GiB

| 项 | 大小 |
|---|---|
| Engram pinned | 188.8 GiB |
| 引擎分片 40 × 6.33 | 253 GiB |
| 非专家 | ~25 GiB |
| **账面合计** | **≈467 GiB** |
| **实测 RSS** | **1064–1116 GB** |
| **缺口** | **≈600 GiB** |

**性质:未点名,不要编解释。** 已排除:
* "钩子前快照扣住 264 GiB" —— **已实测否定**(`p.data` 赋值后底层 storage 变 1 字节,
  原存储确实释放;NOTES §394.3);
* 文件映射 / CoW —— `RssFile` 仅 0.44 GB;
* hugetlb —— `HugePages_Total=0`。

**下一步的量法**(建议在本 patch 落地时一并做):逐层打印
`RssAnon / RssShmem / RssFile` **在"释放前"与"释放后"两个时刻**的增量,
并对同一层的"引擎分片写入字节数"做差 —— 缺口要么落在"释放未归还 OS"
(则加 `malloc_trim(0)` / 调整分配器),要么落在"引擎自身的额外缓冲"(则查 `numa_pool`)。
