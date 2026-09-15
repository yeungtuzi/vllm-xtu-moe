#!/usr/bin/env python
"""Isolate per-engine host memory for ONE layer's MoE engine construction.

Usage:
  XIAOTU_LAYER1_NPZ=fixtures/real_layer1_model.npz python /tmp/measure_layer_mem.py fixture N
  python /tmp/measure_layer_mem.py synth E H I N
"""
import os, sys, gc

REPO = "/home/user/lvllm/vllm-xiaotu-moe"
sys.path.insert(0, REPO)


def smaps_rollup():
    d = {}
    with open("/proc/self/smaps_rollup") as f:
        for ln in f:
            parts = ln.split()
            if len(parts) >= 2 and parts[0].endswith(":"):
                try:
                    d[parts[0][:-1]] = int(parts[1])
                except ValueError:
                    pass
    return d


def status():
    d = {}
    with open("/proc/self/status") as f:
        for ln in f:
            if ":" in ln:
                k, v = ln.split(":", 1)
                d[k] = v.strip()
    return d


def snap(tag):
    r = smaps_rollup()
    s = status()
    print(f"[{tag}] Rss={r.get('Rss',0)/2**10:9.1f} MiB  Pss={r.get('Pss',0)/2**10:9.1f} MiB  "
          f"Anon={r.get('Anonymous',0)/2**10:9.1f} MiB  AnonHuge={r.get('AnonHugePages',0)/2**10:9.1f} MiB  "
          f"VmSize={int(s.get('VmSize','0 kB').split()[0])/2**10:9.1f} MiB  VmRSS={int(s.get('VmRSS','0 kB').split()[0])/2**10:9.1f} MiB")
    return r, s


def make_cfg(m, E, H, I, K=6):
    cfg = m.MOEConfigV2()
    cfg.num_processes = 1; cfg.process_id = 0; cfg.gpu_id = 0
    cfg.has_gate_proj = True; cfg.expert_num = E; cfg.top_k = K
    cfg.hidden_size = H; cfg.intermediate_size = I
    cfg.max_batch_size = 256; cfg.max_num_seqs = 256
    cfg.stride = 32; cfg.group_min_len = 10; cfg.group_max_len = 4096 + 128
    cfg.groupN = 1; cfg.groupK = 32; cfg.activation_type = 0
    return cfg


def run_fixture(npz, neng):
    import numpy as np
    d = np.load(npz)
    E = int(d["E"]); H = int(d["H"]); I = int(d["I"])
    print(f"fixture E={E} H={H} I={I}")
    w13 = np.ascontiguousarray(d["w13"]); w2 = np.ascontiguousarray(d["w2"])

    def f32_to_e8m0(s):
        lg = np.round(np.log2(np.maximum(s, 1e-30))).astype(np.int32) + 127
        return np.clip(lg, 0, 255).astype(np.uint8)

    s13 = np.ascontiguousarray(f32_to_e8m0(d["g13"])); s2 = np.ascontiguousarray(f32_to_e8m0(d["g2"]))
    import xiaotu_moe
    m = xiaotu_moe.load()
    print("variant", xiaotu_moe.__variant__)
    snap("before-engines")
    engines = []
    last = None
    for i in range(neng):
        cfg = make_cfg(m, E, H, I)
        engines.append(m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0))
        r, s = snap(f"engine{i}")
        if last is not None:
            print(f"    delta Rss={ (r['Rss']-last[0]['Rss'])/2**10:8.1f} MiB  "
                  f"delta Anon={ (r['Anonymous']-last[0]['Anonymous'])/2**10:8.1f} MiB  "
                  f"delta VmSize={ (int(s['VmSize'].split()[0])-int(last[1]['VmSize'].split()[0]))/2**10:8.1f} MiB")
        last = (r, s)
    print(f"expected written bytes/layer = (w13+w2) = {(w13.nbytes+w2.nbytes)/2**20:.1f} MiB")
    return engines


def run_synth(E, H, I, neng, K=6, gk=32):
    import numpy as np
    print(f"synth E={E} H={H} I={I} K={K}")
    # source arrays (touch them so RSS is real)
    w13 = np.zeros((E, 2 * I, H // 2), dtype=np.uint8)
    w2 = np.zeros((E, H, I // 2), dtype=np.uint8)
    s13 = np.full((E, 2 * I, H // gk), 127, dtype=np.uint8)
    s2 = np.full((E, H, I // gk), 127, dtype=np.uint8)
    w13.fill(0); w2.fill(0)   # fully fault the source so engine delta is only the engine
    w13b, w2b = w13.nbytes, w2.nbytes
    print(f"source w13={w13b/2**30:.2f} GiB w2={w2b/2**30:.2f} GiB total={(w13b+w2b)/2**30:.2f} GiB")
    import xiaotu_moe
    m = xiaotu_moe.load()
    print("variant", xiaotu_moe.__variant__)
    snap("before-engines")
    engines = []
    last = None
    for i in range(neng):
        cfg = make_cfg(m, E, H, I, K)
        engines.append(m.MOE_MXFP4(cfg, w13, w2, s13, s2, 0, 0))
        r, s = snap(f"engine{i}")
        if last is not None:
            print(f"    delta Rss={ (r['Rss']-last[0]['Rss'])/2**10:8.1f} MiB  "
                  f"delta Anon={ (r['Anonymous']-last[0]['Anonymous'])/2**10:8.1f} MiB  "
                  f"delta AnonHuge={ (r['AnonHugePages']-last[0]['AnonHugePages'])/2**10:8.1f} MiB  "
                  f"delta VmSize={ (int(s['VmSize'].split()[0])-int(last[1]['VmSize'].split()[0]))/2**10:8.1f} MiB")
        last = (r, s)
    print(f"expected written bytes/layer = (w13+w2) = {(w13b+w2b)/2**30:.2f} GiB")
    return engines


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fixture":
        npz = os.environ.get("XIAOTU_LAYER1_NPZ", os.path.join(REPO, "fixtures/real_layer1_model.npz"))
        run_fixture(npz, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif mode == "synth":
        run_synth(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        raise SystemExit("mode must be fixture|synth")
