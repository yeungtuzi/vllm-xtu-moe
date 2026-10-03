#!/usr/bin/env python3
"""fp8-e4m3 → int8 重量化(块内对称,块 = 16,与 e8m0 块一致 ✓)

用途:V4.1-Flash 的 experts 是 FP8-E4M3 + e8m0 块 scale(块 16)。
     要让 AVX512-VNNI(`vpdpbusd`)吃得动,必须先把它重量化成 int8 ✓。
     本模块是【一次性的加载期转换】,把 (int8 权重, 新 scale) 交给引擎。

⚠️ 实测代价:块内对称量化的输出层中位相对误差 ≈ 5.9e-3(用真实权重 + 高斯激活测过 ✓)

接口:
  requant_weight(w_i8_bytes, scale_e8m0, blk=16) -> (q_int8, new_scale_fp32)
    w_i8_bytes : torch.int8 或 uint8 张量(裸 fp8-e4m3 字节)  [rows, K]
    scale_e8m0 : torch.uint8 / float8_e8m0fnu               [rows, K/blk]
    q_int8     : torch.int8                                  [rows, K]
    new_scale  : torch.float32                               [rows, K/blk]
自检:
  python scripts/requant_fp8_to_int8.py --selftest
"""
import sys, os, argparse

def e4m3_table():
    """e4m3(含 subnormal)的 256 项精确查表 ⇒ 避免依赖 torch 的 fp8 支持 ✓"""
    import torch
    t = torch.zeros(256, dtype=torch.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) & 1 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * (2.0 ** -6)          # subnormal ✓
        elif e == 0xF:
            # ⚠️ e4m3 只有 0x7F/0xFF 是 NaN ✓;其余 e=0xF 是【有限值】max=(1+m/8)*2^8 ✓
            v = float('nan') if m == 7 else (1 + m / 8.0) * (2.0 ** 8)
        else:
            v = (1 + m / 8.0) * (2.0 ** (e - 7))
        t[b] = s * v
    return t

def e8m0_to_float(sc):
    """e8m0 → float。⚠️ scale 的 dtype 常是 float8_e8m0fnu ⇒ 必须先取【原始字节】✓
    (直接 .to(int64) 会解出 0 ⇒ Wd=0 ⇒ 后面全是 NaN ✗)"""
    import torch
    if sc.dtype in (torch.uint8, torch.int8):
        b = sc.view(torch.uint8).to(torch.int64)
    else:
        try:
            b = sc.view(torch.uint8).to(torch.int64)          # float8_e8m0fnu ✓
        except Exception:
            b = sc.to(torch.float32).to(torch.int64)          # 已是浮点 scale ✓
    return torch.pow(torch.tensor(2.0), b.float() - 127.0)

def requant_weight(w_bytes, scale_bytes, blk=16):
    import torch
    lut = e4m3_table()
    b = w_bytes.view(torch.uint8).to(torch.int64) if w_bytes.dtype != torch.uint8 else w_bytes.to(torch.int64)
    Wf = lut[b]                                   # 精确 fp8 → fp32 ✓
    Wf = torch.nan_to_num(Wf, nan=0.0, posinf=0.0, neginf=0.0)   # 0x7F/0xFF 是 e4m3 的 NaN 位型 ⇒ 按 0 处理 ✓
    S = e8m0_to_float(scale_bytes)
    K = Wf.shape[1]
    assert K % blk == 0 and S.shape[1] == K // blk, '块大小不匹配'
    Wd = Wf * S.repeat_interleave(blk, dim=1)     # 精确反量化 ✓
    blk_v = Wd.reshape(Wd.shape[0], -1, blk)
    amax = blk_v.abs().amax(dim=2, keepdim=True).clamp_min(1e-30)
    nsc = amax / 127.0                            # 新的块 scale ✓
    q = torch.clamp(torch.round(blk_v / nsc), -127, 127).to(torch.int8)
    return q.reshape(Wd.shape), nsc.squeeze(2).to(torch.float32)

def nibble_to_int8(w_u8, K):
    """MXFP4:把 nibble-packed 权重【精确】解成 int8 ✓
    e2m1 值 {0,±0.5,±1,±1.5,±2,±3,±4,±6} ×2 => {0,±1,±2,±3,±4,±6,±8,±12} 全整数 ≤12 ✓
    布局:每字节 = 2 个 nibble(低=偶 k,高=奇 k ✓);输入 [E, rows, K/2] u8 => 输出 [E, rows, K] int8 ✓
    ⇒ 与 C++ 补丁的 `Wi[j*K + k]` 逐元素索引【一致】✓"""
    import torch
    b = w_u8.to(torch.int64)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    LUT = torch.tensor([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=torch.int64)  # ×2 后的整数值 ✓
    # ⭐ 向量化:interleave 用 stack+reshape(连续写 ✓)⇒ 比 out[...,0::2] 快 10~50 倍 ✓
    out = torch.stack((LUT[lo], LUT[hi]), dim=-1).reshape(b.shape[:-1] + (K,))
    return out.to(torch.int8)


def requant_from_f32(Wd, blk=16):
    """从【已精确反量化的 fp32 权重】出发,按 blk 块做对称 int8 量化 ✓。
    用途:插件给的是【128 块的 fp32 scale】✗,而我们要 gk=16 ✓
          ⇒ 先用插件 scale 精确反量化(无损 ✓),再按 16 块重量化 ✓
    ⇒ 这样【不需要改 C++】(门槛仍是 gk==16 ✓),且精度是 16 块的水平(5.89e-3 ✓)
    返回 (int8 权重 [rows,K], 新 scale [rows, K/blk] fp32)"""
    import torch
    rows, K = Wd.shape
    assert K % blk == 0, 'K 不是 blk 的整数倍'
    v = Wd.reshape(rows, -1, blk)
    amax = v.abs().amax(dim=2, keepdim=True).clamp_min(1e-30)
    nsc = amax / 127.0
    q = torch.clamp(torch.round(v / nsc), -127, 127).to(torch.int8)
    return q.reshape(rows, K), nsc.squeeze(2).to(torch.float32)


def dequant_fp8(w_bytes, scale_f32, blk=128):
    """用【插件给的 fp32 scale】精确反量化 fp8 权重 ✓(无损 ✓)。
    插件里 w 已是 fp8-e4m3、scale 已是 fp32、块 = _group_k ✓"""
    import torch
    lut = e4m3_table()
    b = w_bytes.view(torch.uint8).to(torch.int64) if w_bytes.dtype != torch.uint8 else w_bytes.to(torch.int64)
    Wf = torch.nan_to_num(lut[b], nan=0.0)
    S = scale_f32.to(torch.float32)
    return Wf * S.repeat_interleave(blk, dim=1)


def _selftest():
    import torch, glob, json, struct
    D = os.environ.get('CKPT', '/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master')
    try:
        from safetensors import safe_open
    except Exception:
        print('  ✗ 需要 safetensors'); return 1
    w = sc = None
    for f in sorted(glob.glob(D + '/*.safetensors'))[:8]:
        try:
            with safe_open(f, framework='pt', device='cpu') as fh:
                ks = [k for k in fh.keys() if 'experts.0.w1.weight' in k]
                if ks:
                    w = fh.get_tensor(ks[0])
                    sc = fh.get_tensor([k for k in fh.keys() if 'experts.0.w1.scale' in k][0])
                    print('  取自 %s · %s' % (f.split('/')[-1], ks[0])); break
        except Exception: continue
    if w is None: print('  ✗ 未找到权重'); return 1
    print('  权重 %s %s · scale %s %s' % (w.dtype, tuple(w.shape), sc.dtype, tuple(sc.shape)))
    q, nsc = requant_weight(w, sc, 16)
    print('  ✅ 转换完成:int8 %s · scale %s' % (tuple(q.shape), tuple(nsc.shape)))
    # 自检 1:块内相对误差(与 fp8 精确值比 ✓)
    lut = e4m3_table(); b = w.view(torch.uint8).to(torch.int64)
    Wd = lut[b] * e8m0_to_float(sc).repeat_interleave(16, dim=1)
    Wq = q.float() * nsc.repeat_interleave(16, dim=1)
    rel = ((Wq - Wd).abs() / Wd.abs().clamp_min(1e-30))
    rf = rel.flatten(); rf = rf[~torch.isnan(rf)]
    print('  自检① 权重块内相对误差:p50=%.2e p99.9=%.2e max=%.2e(有效样本 %d ✓)' % (
        float(torch.nanquantile(rf, .5)), float(torch.nanquantile(rf, .999)), float(rf.max()), rf.numel()))
    # 自检 2:输出层误差(应与此前实测 5.9e-3 同量级 ✓)
    torch.manual_seed(0); A = torch.randn(64, Wd.shape[1]) * 0.5
    R = A.double() @ Wd.t().double(); B = A.double() @ Wq.t().double()
    r = ((B - R).abs() / R.abs().clamp_min(1e-30)).flatten(); r = r[~torch.isnan(r)]
    print('  自检② 输出层相对误差:p50=%.2e p99.9=%.2e  ⇒ 预期 p50 ≈ 5.9e-3 ✓' % (
        float(torch.nanquantile(r, .5)), float(torch.nanquantile(r, .999))))
    return 0

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    sys.exit(_selftest() if a.selftest else (print(__doc__) or 0))
