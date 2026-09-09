#!/usr/bin/env bash
# Collect measured hardware bandwidth into report/hw_bandwidth.json.
# Run on an idle machine (CPU MoE / loading jobs distort STREAM badly).
set -euo pipefail
cd "$(dirname "$0")"
echo "loadavg_before=$(cat /proc/loadavg)" > hw_bandwidth.txt

run() { echo "$*" | tee -a hw_bandwidth.txt; }

NODE=$(numactl --cpunodebind=1 --membind=1 ./stream 200000000 24 | tail -1)
SOCK=$(numactl --cpunodebind=0-3 --membind=0-3 ./stream 400000000 48 | tail -1)
ALL=$(numactl --interleave=all ./stream 400000000 192 | tail -1)
run "$NODE"; run "$SOCK"; run "$ALL"

PCI=$(python pcie_bw.py)
echo "$PCI" | tee -a hw_bandwidth.txt

python - "$NODE" "$SOCK" "$ALL" "$PCI" <<'PY'
import json, re, sys
node, sock, allm, pci = sys.argv[1:5]
def tri(s):
    return float(re.search(r"Triad ([\d.]+)", s).group(1))
pcie = {}
for line in pci.strip().splitlines():
    m = re.match(r"(pinned H2D|pinned D2H|pageable H2D)\s+([\d.]+)", line.strip())
    if m:
        pcie[m.group(1)] = float(m.group(2))
hbm = 0.0
for line in pci.strip().splitlines():
    m = re.match(r"HBM D2D copy\s+([\d.]+)", line.strip())
    if m:
        hbm = float(m.group(1))
d = {"stream": {"1 NUMA node (24c)": tri(node),
                "1 socket (48c)": tri(sock),
                "all 192 threads": tri(allm)},
     "pcie": pcie, "hbm_d2d": hbm}
json.dump(d, open("hw_bandwidth.json", "w"), indent=1)
print(json.dumps(d, indent=1))
PY
