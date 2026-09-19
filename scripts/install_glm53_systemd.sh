#!/usr/bin/env bash
# 把 deploy/systemd/glm53.service 装成 **用户级** systemd 单元(不需要 root)。
#
#   bash scripts/install_glm53_systemd.sh            # 只安装 + daemon-reload(不启动、不 enable)
#   bash scripts/install_glm53_systemd.sh --enable   # 额外:enable(以后开机自起;若报要 linger,见下)
#
# 装完之后的日常:
#   systemctl --user start glm53        # 起(~5 min 后才 READY,用 curl 验)
#   systemctl --user status glm53
#   journalctl --user -u glm53 -n 200 -f
#   systemctl --user stop glm53         # 停
#
# 开机自起需要 linger(用户注销后仍允许用户服务运行):
#   sudo loginctl enable-linger "$USER"        # ← 需要 sudo,由你决定要不要开
#
# ⚠️ 手动(no)hup 模式与本单元**互斥**(同端口 8070、同 GPU 0/1):
#   切到 systemd 之前先 `kill "$(cat logs/glm53_prod.pid)"`,并确认 `nvidia-smi` 显存归零。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/deploy/systemd/glm53.service"
DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
DST="$DST_DIR/glm53.service"

[ -f "$SRC" ] || { echo "找不到 $SRC" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "这台机器没有 systemctl,只能用 nohup 模式" >&2; exit 1; }

mkdir -p "$DST_DIR"
cp "$SRC" "$DST"
echo "[install] $DST"

# 如果 8070 上已经有手动起的实例,提醒(不自动杀:可能是正在服务的生产实例)
if curl -sf -m 3 "http://127.0.0.1:8070/v1/models" >/dev/null 2>&1; then
  echo "[install] ⚠️ 8070 现在**已经有服务在跑**(手动模式?)。"
  echo "          要切到 systemd,先停掉它:kill \"\$(cat $ROOT/logs/glm53_prod.pid)\" 并等显存归零。"
fi

systemctl --user daemon-reload
echo "[install] daemon-reload 完成"

if [ "${1:-}" = "--enable" ]; then
  systemctl --user enable glm53
  echo "[install] 已 enable(开机自起还要求 linger:sudo loginctl enable-linger $USER)"
fi

cat <<EOF

[install] 下一步:
  systemctl --user start glm53     # 起(~5 min;就绪判定:curl -sf http://127.0.0.1:8070/v1/models)
  bash scripts/glm53_status.sh     # 一页纸状态(journal 模式下 pidfile 可能不存在,属正常)
  journalctl --user -u glm53 -f    # 日志
EOF
