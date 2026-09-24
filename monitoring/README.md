# monitoring —— 自建 Grafana / Prometheus 监控栈(源码)

> 这是 8070 生产栈的**监控侧源码**(仪表盘、provisioning、抓取配置、自制导出器与网页)。
> 之前它只存在于 `/home/user/lvllm/monitoring/`(**未被版本库跟踪**),本目录是入库的**可信副本**。
> 台账依据:`docs/EXPERIMENTS.md` **B204–B208**;运行说明见 `dev-docs/HANDOFF.md` §4。

## 1. 组件与端口

| 组件 | 版本 | 端口 | 说明 |
|---|---|---|---|
| **Grafana** | 11.4.0 | **3000** | 唯一看板入口 `http://127.0.0.1:3000/d/dsh-overview`(uid `dsh-overview`) |
| **Prometheus** | **2.45.6 LTS** ⚠️ | 9090 | 4 个抓取目标(vllm / lmcache / node / xtu textfile) |
| **node_exporter** | 1.8.2 | 9100 | 主机 CPU/内存 |
| **xtu_exporter**(自写) | — | 9100 textfile | 逐卡 GPU / 逐 NUMA / 每物理核 |
| **coremap**(自写) | — | 8787 | 核心地图 HTML + PNG,每 5s 自刷新 |
| **LMCache 服务端** | fork 0.5.5 | 5555 ZMQ / 8080 HTTP | L1 内存 + L2 磁盘前缀缓存 |

看板结构:vLLM 报表在最上(6 张头条卡)→ 主机报表(11 张头条)→ 细节默认折叠(6 个折叠行 / 26 个面板),
含 **CPU 核心地图** 与 **GPU 矩阵**。旧的两张(vllm-lmcache / host-numa-gpu-cpu)已退役,备份在
`retired-dashboards/`。

## 2. 目录内容

| 路径 | 作用 |
|---|---|
| `dashboards/dsh-overview.json` | **当前唯一在用的看板**(已合并旧两张) |
| `retired-dashboards/*.json` | 退役看板备份(**已移出 provisioning 扫描目录**) |
| `grafana/grafana.ini` | Grafana 主配置(匿名 Admin、embedding、sanitize 关闭) |
| `grafana/provisioning/dashboards/dash.yml` | 看板 provider(注意 `path` 指向运行目录) |
| `grafana/provisioning/datasources/prom.yml` | Prometheus 数据源(默认) |
| `prometheus/prometheus.yml` | 抓取配置:5s 间隔,vllm/node/lmcache 三目标 |
| `textfile_exporter.py` | 自写 textfile 导出器(GPU / NUMA / 每核心) |
| `coremap_png.py` | CPU 核心地图 PNG(PIL 直绘;NUMA→CCD→每核) |
| `gpumap_png.py` | GPU 矩阵 PNG(按占用者切块 + SM/显存/温度/功耗) |
| `web/serve.py`、`web/coremap.html` | coremap 静态服务(:8787,须发 `Cache-Control: no-store`) |
| `shot/` | Playwright 截图小工具(看板取证用) |

## 3. 运行位置与"源码"的关系

* **运行目录**仍是 `/home/user/lvllm/monitoring/`(Grafana/Prometheus 二进制、`data/`、日志都在那里);
  `grafana.ini` 与 `dash.yml` 里的绝对路径指向它。
* 本目录是**版本化的源码副本**;改动仪表盘/配置后,应同步回运行目录并重启对应受管进程
  (`bash scripts/proc.sh ...`,见 `dev-docs/HANDOFF.md` §3)。
* **不入库**:Grafana / Prometheus / node_exporter 二进制与压缩包、`data/`(WAL、`grafana.db`)、
  日志、`node_modules/`、生成的 `*.png`、运行时 `textfile/*.prom`。见本目录 `.gitignore`。

## 4. 坑(务必记住)

* **Prometheus 3.1(Go 1.23)在本机连 `--version` 都段错误** ✗ ⇒ 必须用 **2.45.6 LTS** ✓
* **Grafana 的 text 面板会把 `<iframe>` 过滤掉** ✗(即使 `disable_sanitize_html=true` + 重启)⇒ 用 **`<img>`** ✓
* **PIL 的 DejaVu 无中文字形** ✗ ⇒ PNG 里只能用 ASCII ✓
* 静态服务器必须发 **`Cache-Control: no-store`** ✓,否则刷新看不到新图 ✗
* **不要用降采样的整页截图判断颜色** ✗(会把绿色糊成蓝灰);用**原图或像素统计** ✓
* 进程纪律:**禁止** `pkill -f` / 模式匹配清理(会命中自己与 LMCache 服务)⇒ 只用 PID 文件 ✓
