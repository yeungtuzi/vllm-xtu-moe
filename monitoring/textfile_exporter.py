#!/usr/bin/env python3
"""Write GPU (per card) and NUMA (per node) metrics as a Prometheus textfile.

Runs a loop and atomically replaces the .prom file so node_exporter's textfile
collector always reads a complete file. GPU data comes from nvidia-smi; NUMA
data comes from /sys/devices/system/node.
"""
import os, subprocess, time, tempfile, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/lvllm/monitoring/textfile/xtu.prom"
INTERVAL = float(os.environ.get("XTU_EXPORT_INTERVAL", "5"))
Q = "index,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw"


def gpu_lines():
    try:
        raw = subprocess.run(
            ["nvidia-smi", f"--query-gpu={Q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        ).stdout.strip()
    except Exception:
        return []
    out = []
    for row in raw.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 8:
            continue
        idx, name, mused, mtotal, util, mutil, temp, power = parts[:8]
        lbl = f'gpu="{idx}",name="{name}"'
        def num(x):
            try: return float(x)
            except Exception: return None
        for metric, val, scale in (
            ("xtu_gpu_memory_used_bytes", num(mused), 1024 * 1024),
            ("xtu_gpu_memory_total_bytes", num(mtotal), 1024 * 1024),
            ("xtu_gpu_utilization_percent", num(util), 1),
            ("xtu_gpu_memory_utilization_percent", num(mutil), 1),
            ("xtu_gpu_temperature_celsius", num(temp), 1),
            ("xtu_gpu_power_watts", num(power), 1),
        ):
            if val is not None:
                out.append(f"{metric}{{{lbl}}} {val * scale}")
        mu, mt = num(mused), num(mtotal)
        if mu is not None and mt:
            out.append(f'xtu_gpu_memory_used_percent{{{lbl}}} {100.0 * mu / mt}')
    return out


def gpu_proc_lines():
    """Per-process GPU memory, so a card's VRAM can be shown as one tile per owner."""
    try:
        uuids = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception:
        return []
    idx_of = {}
    for row in uuids.splitlines():
        parts = [x.strip() for x in row.split(",")]
        if len(parts) == 2:
            idx_of[parts[1]] = parts[0]
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception:
        return []
    out = []
    for row in raw.splitlines():
        parts = [x.strip() for x in row.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid, mem = parts[0], parts[1], parts[2]
        gpu = idx_of.get(uuid, "?")
        try:
            with open(f"/proc/{pid}/comm") as fh:
                name = fh.read().strip()
        except Exception:
            name = "?"
        try:
            out.append(f'xtu_gpu_proc_memory_bytes{{gpu="{gpu}",pid="{pid}",name="{name}"}} '
                       f'{float(mem) * 1024 * 1024}')
        except Exception:
            pass
    return out


def gpu_extra_lines():
    """Clocks, fan and power limit per card, for the indicator tiles."""
    try:
        raw = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,clocks.sm,clocks.mem,fan.speed,power.limit,power.draw,"
             "utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception:
        return []
    out = []
    for row in raw.splitlines():
        parts = [x.strip() for x in row.split(",")]
        if len(parts) < 7:
            continue
        idx, csm, cmem, fan, plimit, pdraw, util = parts[:7]
        def num(x):
            try:
                return float(x)
            except Exception:
                return None
        for metric, val in (("xtu_gpu_clock_sm_mhz", num(csm)),
                            ("xtu_gpu_clock_mem_mhz", num(cmem)),
                            ("xtu_gpu_fan_percent", num(fan)),
                            ("xtu_gpu_power_limit_watts", num(plimit))):
            if val is not None:
                out.append(f'{metric}{{gpu="{idx}"}} {val}')
    return out


def numa_lines():
    """Per-NUMA-node memory, using free(1) semantics.

    The kernel's node MemUsed is MemTotal - MemFree, which counts reclaimable
    page cache as used; that made a host with 480 GiB used and 1.0 TiB of cache
    look 99 percent full. Available therefore subtracts the file pages and
    SReclaimable, so the numbers line up with free -h.
    """
    base = "/sys/devices/system/node"
    out = []
    if not os.path.isdir(base):
        return out
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("node"):
            continue
        node = entry[4:]
        info = {}
        try:
            with open(os.path.join(base, entry, "meminfo")) as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 4:
                        info[parts[2].rstrip(":")] = float(parts[3]) * 1024.0
        except Exception:
            continue
        total = info.get("MemTotal")
        if not total:
            continue
        free = info.get("MemFree", 0.0)
        file_pages = info.get("Active(file)", 0.0) + info.get("Inactive(file)", 0.0)
        reclaimable = info.get("SReclaimable", 0.0)
        cache = file_pages + reclaimable
        shared = info.get("Shmem", 0.0)
        # free-style: "used" excludes reclaimable page cache AND shmem/tmpfs,
        # which free reports separately as shared. used + shared + cache + free
        # then adds up to MemTotal, so the stack reconciles with free -h.
        available = free + cache
        consumed = max(0.0, total - available - shared)
        out += [
            f'xtu_numa_mem_total_bytes{{node="{node}"}} {total}',
            f'xtu_numa_mem_free_bytes{{node="{node}"}} {free}',
            f'xtu_numa_mem_cache_bytes{{node="{node}"}} {cache}',
            f'xtu_numa_mem_shared_bytes{{node="{node}"}} {shared}',
            f'xtu_numa_mem_available_bytes{{node="{node}"}} {available + shared}',
            f'xtu_numa_mem_used_bytes{{node="{node}"}} {consumed}',
            f'xtu_numa_mem_used_percent{{node="{node}"}} {100.0 * consumed / total}',
            f'xtu_numa_mem_available_percent{{node="{node}"}} {100.0 * (available + shared) / total}',
        ]
        try:
            with open(os.path.join(base, entry, "cpulist")) as fh:
                spec = fh.read().strip()
            count = 0
            for part in spec.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    count += int(b) - int(a) + 1
                elif part:
                    count += 1
            out.append(f'xtu_numa_cpu_count{{node="{node}"}} {count}')
        except Exception:
            pass
    return out


def _l3_group(cpu):
    """CCD/CCX index for a cpu: the L3 cache group it shares (sysfs index3)."""
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list") as fh:
            spec = fh.read().strip()
        return spec          # e.g. "0-7" -> one CCD
    except Exception:
        return None


def core_topology():
    """Map physical core id -> (cpu list, numa node, ccd)."""
    cores = {}
    base = "/sys/devices/system/cpu"
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("cpu") or not entry[3:].isdigit():
            continue
        cpu = int(entry[3:])
        try:
            with open(os.path.join(base, entry, "topology", "core_id")) as fh:
                core = int(fh.read().strip())
            with open(os.path.join(base, entry, "topology", "physical_package_id")) as fh:
                pkg = int(fh.read().strip())
        except Exception:
            continue
        key = (pkg, core)
        cores.setdefault(key, []).append(cpu)
    # numa node per cpu (from node*/cpulist)
    cpu_node = {}
    nbase = "/sys/devices/system/node"
    if os.path.isdir(nbase):
        for nd in os.listdir(nbase):
            if not nd.startswith("node"):
                continue
            try:
                with open(os.path.join(nbase, nd, "cpulist")) as fh:
                    spec = fh.read().strip()
            except Exception:
                continue
            for part in spec.split(","):
                if "-" in part:
                    a, b = part.split("-")
                    rng = range(int(a), int(b) + 1)
                elif part:
                    rng = [int(part)]
                else:
                    continue
                for c in rng:
                    cpu_node[c] = nd[4:]
    out = {}
    l3_seen = {}
    for k, v in cores.items():
        cpu0 = sorted(v)[0]
        l3 = _l3_group(cpu0)
        if l3 is not None and l3 not in l3_seen:
            l3_seen[l3] = str(len(l3_seen))
        out[k] = (v, cpu_node.get(cpu0, "?"), l3_seen.get(l3, "?"))
    return out


def read_stat():
    busy, total = {}, {}
    with open("/proc/stat") as fh:
        for line in fh:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            idx = int(parts[0][3:])
            vals = [int(x) for x in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            tot = sum(vals)
            busy[idx] = tot - idle
            total[idx] = tot
    return busy, total


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    topo = core_topology()
    prev = None
    while True:
        cur = read_stat()
        core_lines = []
        if prev is not None:
            pbusy, ptot = prev
            cbusy, ctot = cur
            for (pkg, core), (cpus, node, ccd) in sorted(topo.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                best = None
                for c in cpus:
                    if c in cbusy and c in pbusy and ctot.get(c, 0) > ptot.get(c, 0):
                        d = (ctot[c] - ptot[c]) or 1
                        u = 100.0 * (cbusy[c] - pbusy[c]) / d
                        best = u if best is None else max(best, u)
                if best is not None:
                    core_lines.append(
                        f'xtu_core_utilization_percent{{core="{core}",node="{node}",'
                        f'package="{pkg}",ccd="{ccd}",threads="{len(cpus)}"}} '
                        f'{max(0.0, min(100.0, best))}')
        prev = cur
        lines = (["# HELP xtu_gpu_memory_used_bytes GPU memory used.",
                  "# TYPE xtu_gpu_memory_used_bytes gauge"] + gpu_lines() + numa_lines() +
                 ["# HELP xtu_core_utilization_percent Per physical core CPU utilization.",
                  "# TYPE xtu_core_utilization_percent gauge"] + core_lines +
                 gpu_proc_lines() + gpu_extra_lines())
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(OUT), prefix=".xtu", suffix=".prom")
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, OUT)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
