# 多模型 × 多副本:统一入口、模型选择与负载均衡(规划)

> **状态:规划(设计已定,实现与验证未做)**。本文只描述架构与前置条件,
> **尚未运行任何多实例实验** —— 待验证项列在 §8,届时按清单逐条落实。

## 1. 目标

同时提供我们**已测试的 3 个模型**:

| 对外模型名(`/v1/models` 里唯一暴露的)| 说明 |
|---|---|
| `DeepSeek-V4.1-Flash` | TP=2,768K,GPU 预填 + dspark + LMCache |
| `GLM-5.3-Flash` | TP=2 |
| `mimo26` | MiMo-V2.6-Flash-RL,TP=2 |

每个模型部署 **M 份副本**(总实例数 **N = 3M**),要求:

* 对外只有**一个端口 + 一套 API Key**;
* 用户**按模型名选择**;
* 后台在**同名副本之间自动负载均衡**;
* **同一模型的 M 份共享同一份前缀/KV 缓存**(配置一致 ⇒ 缓存的 KV 可互换)。

## 2. 架构

```
用户 / DeepSeek Harness
      │  统一端口(如 4000)+ 统一 API Key
      ▼
┌──────────────────────────────────────────────┐
│ 网关:LiteLLM Proxy                          │
│  model_name: DeepSeek-V4.1-Flash ⇒ {A1…AM}  │  ← 同名多条 = 一个"模型组"
│  model_name: GLM-5.3-Flash        ⇒ {B1…BM}  │
│  model_name: mimo26               ⇒ {C1…CM}  │
└──────────────────────────────────────────────┘
      │ 各副本只绑 127.0.0.1(不直接对外)
      ▼
 vLLM 副本 A1…CM(端口 8071…807N,全部 `--enable-prefix-caching`)
      │  lmcache.mp.host/port(所有副本指向同一台)
      ▼
 一台 **LMCache 服务端**(L1 = 主机内存,跨副本共享;L2 = 磁盘,跨副本共享)
```

**关键点**:网关按 `model_name` 归组 ⇒ `/v1/models` 只暴露 **3 个模型**,`M` 对用户完全透明。

## 3. 网关配置骨架(LiteLLM)

每个模型写 **M 条同名条目**,每条一个 `api_base`:

```yaml
model_list:
  - model_name: DeepSeek-V4.1-Flash
    litellm_params:
      model: openai/DeepSeek-V4.1-Flash
      api_base: http://127.0.0.1:8071/v1
      api_key: os.environ/XTU_REPLICA_A1_KEY
  - model_name: DeepSeek-V4.1-Flash          # 同名 ⇒ 自动进同一模型组
    litellm_params:
      model: openai/DeepSeek-V4.1-Flash
      api_base: http://127.0.0.1:8073/v1
      api_key: os.environ/XTU_REPLICA_A2_KEY
  # … 共 M 条;GLM-5.3-Flash 与 mimo26 同理

router_settings:
  routing_strategy: least-busy   # 长上下文/长预填场景比随机更稳
  num_retries: 2                 # 副本故障自动换一个
  timeout: 1800                  # 长预填必须放宽(默认 30s 会误杀)
  allowed_fails: 2

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

* 负载均衡策略可选 `simple-shuffle`(默认)/ `least-busy` / `latency-based-routing` 等
  —— 见 [LiteLLM · Load Balancing](https://docs.litellm.ai/docs/proxy/load_balancing)
* 对外签发**虚拟密钥**(按人/按团队,可设预算与限速)
  —— 见 [LiteLLM · Virtual Keys](https://docs.litellm.ai/docs/proxy/virtual_keys)
* **务必**:各副本 vLLM 只绑 `127.0.0.1`,对外只开网关端口。

## 4. 共享缓存:设计、前提与收益

**共享的载体是 LMCache**(它存的就是 **KV 张量**):L1 在服务端内存、L2 在磁盘 ⇒ **跨副本可见**。
vLLM 自己那份 **GPU** 前缀缓存只能在各自显存内,**无法跨进程共享**;跨副本未命中会落到**共享 L1**,
代价远低于重新 prefill。

要真正拿到共享,**四条件必须同时成立**:

| # | 条件 | 原因 |
|---|---|---|
| 1 | LMCache 服务端**只有一台**,所有副本指向同一 `lmcache.mp.host/port` | 共享的前提 |
| 2 | **`CHUNK_SIZE` 所有副本相同**(三个模型都用 **2176**) | 对象键含 chunk 尺寸,不一致则互不命中 |
| 3 | **KV dtype / TP / `max_model_len` 一致**(`fp8_ds_mla`、TP=2、768K) | 存的 KV 必须可互换 |
| 4 | 模型路径与名称不变 | 参与对象键 |

**收益**:**缓存只存一份**(而非 M 份)⇒ 命中率随 M **放大**,内存开销**不随 M 线性增长**。
⇒ **L1 应尽量大**(主机内存充足时,`L1_GB` 可设数百 GiB),L2 放在较快的盘上。

## 5. 端口与命名规范(建议)

| 用途 | 端口 |
|---|---|
| 网关对外 | **4000** |
| vLLM 副本(只绑 127.0.0.1) | **8071…807N** |
| LMCache 服务端 | **5555**(ZMQ)/ **8080**(HTTP 巡检) |

* 对外模型名**保持与现在一致**(`DeepSeek-V4.1-Flash` / `GLM-5.3-Flash` / `mimo26`)⇒ 客户端**无需改模型 id**
* 各副本用 `scripts/proc.sh spawn` 受管启动(`TAG`/`PORT`/`GPUS` 已支持)

## 6. 客户端接入(以 DeepSeek Harness 为例)

只改一行:`baseURL` 从副本端口改为**网关端口**(如 `http://127.0.0.1:4000/v1`);
模型 id **不变**;`/v1/models` 将只列 **3 个模型**。
(DSH 侧模型条目仍需按 `input` / `reasoningEfforts` 等字段声明能力 —— 见本仓 `docs/MODEL_GUIDES.md`。)

## 7. 容量规划

* **显存**:三模型均为 **TP=2** ⇒ **每个副本需 2 张卡** ⇒ `N=3M` 需 ≈ **6M 张 A100**。
* **主机内存**:L1 **只一份**(共享红利);每个副本另有自己的进程开销。
* **单点**:LMCache 服务端是单点 —— 若挂,全体**降级为无共享缓存**(仍可服务,只是变慢)。
  需要高可用就得多台 + `n_servers>1`(连接器支持,见下)。

## 8. ⚠️ 待验证清单(**本阶段未做任何实验**)

- [ ] **N 个 vLLM 实例 → 1 台 LMCache 服务端** 的**跨实例命中**(设计上支持:连接器有
      `ranks_per_server = world_size // n_servers`、`vllm_worker_id=pc.rank`,并明确 `lmcache.mp.host/port`;
      但**我们实测过的只有 1 个实例**)
- [ ] 共享缓存下各 `routing_strategy` 的**实际效果**(命中率 / TTFT 分布)
- [ ] LMCache 服务端在 M 个副本下的**注册与并发行为**(此前日志只出现过 `worker slot 0 of 1` = 单实例)
- [ ] L2 磁盘容量与淘汰策略在长期运行下的表现
- [ ] 备选方案评估:**prefix-aware 路由**(vLLM 官方 Production Stack 具备,
      见 [Prefix Aware Routing 教程](https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.6/_sources/tutorials/prefixaware.rst);
      **但其形态是 K8s/Helm**,单机不自带 ⇒ 目前不作为首选)

## 9. 落地步骤(将来)

1. 装网关:`pip install "litellm[proxy]"`(外网需代理)
2. 写 `gateway/config.yaml`(§3 骨架,密钥走环境变量)
3. 起 **一台** LMCache 服务端(`CHUNK_SIZE=2176`),再按"模型 × M"起副本(建议加一个
   `scripts/serve_fleet.sh`,自动分配端口与 GPU,全部 `proc.sh` 受管)
4. 冒烟:`/v1/models` **只列 3 个模型**;连续同前缀请求观察 `model_id` 在 M 个副本间轮转,
   且 **TTFT 仍为 ~1 s 级**(说明共享缓存生效)
5. 按 §8 逐条验证后再投产
