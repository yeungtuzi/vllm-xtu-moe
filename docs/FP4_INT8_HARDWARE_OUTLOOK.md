# 4-bit Inference Formats and CPU-First MoE Serving: A Hardware Outlook

> **What this document is**: a factual, source-cited reading of the *public* hardware roadmaps
> concerning 4-bit inference formats, plus what those facts imply for a CPU-first MoE serving
> engine. It is **not** a product plan.
>
> **Rules used throughout**: every external claim carries a source; *shipping*, *announced* and
> *rumored* are labelled separately; anything we could not verify is marked **unverified**.

---

## 1. Summary

1. **The 4-bit format question is settled, and the answer is floating-point, not integer.**
   The industry's 4-bit format is **FP4 in the microscaling (MX) style**: E2M1 elements with a
   shared block scale. **INT4** (uniform integer with per-group scale/zero-point) is a previous
   generation of the same idea and has receded from the server roadmaps.
2. **No announced CPU provides native 4-bit matrix multiplication.** CPU matrix extensions
   either stop at 8-bit or accept 4-bit input only for *conversion* — so the practical pattern is
   **4-bit storage, 8-bit compute**.
3. Therefore a CPU-first MoE engine should **keep the model's FP4 format** and treat the
   4-bit→8-bit mapping as a **swappable layer**. That is the design this project follows.
4. The two CPU variables actually worth tracking are **memory bandwidth** and **per-CCD last-level
   cache capacity** — not new 4-bit instructions.

---

## 2. The landscape, with sources

### 2.1 Datacenter GPUs support FP4; INT4 is gone

NVIDIA's 5th-generation tensor-core integer kinds are `kind::i8` and `kind::ti16` only; the 4-bit
kinds are `kind::mxf4` (MXFP4) and `kind::mxf4nvf4` (NVFP4). There is no `i4`.
— [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
NVIDIA's CUTLASS documentation lists exactly one 4-bit type, `float_e2m1_t`.
— [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
NVIDIA's own per-generation precision tables name NVFP4 and INT8, never INT4.
— [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)

AMD's MI355X supports native MXFP4/MXFP6; **NVFP4 is not native and is requantised to MXFP4**.
— [ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/nvfp4-to-mxfp4/README.html)
Intel's announced 4-bit GPU direction is FP4 (Crescent Island / Xe3P).
— [TechPowerUp](https://www.techpowerup.com/351901/intel-details-crescent-island-graphics-32-xe3p-cores-up-to-480-gb-lpddr5x-memory)

### 2.2 x86 CPUs: 8-bit is the floor, 4-bit is convert-only

**AMD, ACE (AI Compute Extensions)** — a joint AMD/Intel x86 matrix specification — is the only
announced CPU matrix engine. Its native matmul is **MXFP8 / MXINT8 / BF16 / INT8 with E8M0 block
scales**; the specification states that **"MX FP4 and MX FP6 have dedicated convert operations"**,
i.e. 4-bit is not a native matmul type. INT4 does not appear.
— [ACE specification](https://x86ecosystem.org/wp-content/uploads/2026/06/ACE_v1_Specification_public.pdf)

**Intel Xeon** progression is AMX-INT8 → AMX-BF16 → AMX-FP16 → (announced for Diamond Rapids)
AMX-FP8. No FP4 and no INT4 matrix instruction is announced on the Xeon roadmap; AMX-FP8 is
8-bit (BF8 × HF8).
— [Intel ISA extensions reference](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-architecture-instruction-set-extensions-programming-reference.html)

**AMD EPYC** adds AVX512-FP16/BF16, AVX-IFMA, AVX-NE-CONVERT, AVX-VNNI-INT8 and AVX512_BMM on
Zen 6 — none of which is 4-bit.
— [AMD doc #69192](https://docs.amd.com/v/u/en-US/69192-PUB)

### 2.3 ARM specifies 6-bit before 4-bit

The published A-profile ISA specifies INT8/FP8/BF16/FP16 and, in Armv9.7-A, **MXFP6** — but no
FP4 and no INT4 as a matrix datatype. INT4 exists above the ISA, in ARM's own KleidiAI
micro-kernels, which **dequantise into INT8/FP16** before computing.
— [Arm 2025 extensions](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-a-profile-architecture-developments-2025)
— [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai/-/merge_requests/153)

### 2.4 Apple documents both — and the order is instructive

Apple shipped **INT4** tensor data types in iOS/macOS 26.4, and has announced **MXFP4**
(`metalFloat4e2m1` with `metalFloat8ue8m0` block-32 scales) for 27.0. Its MSL specification
defines the MXFP4 tensor with an explicit `tensor_blockwise<..., ue8m0, 32, 1>` layout, i.e. OCP
MXFP4.
— [MTLTensorDataType.int4](https://developer.apple.com/documentation/metal/mtltensordatatype/int4)
— [metalFloat4e2m1](https://developer.apple.com/documentation/metal/mtltensordatatype/metalfloat4e2m1)

**Reading:** INT4 arrived first and MXFP4 followed. The migration direction is
**int4 → FP4**, not the reverse.

### 2.5 Summary table

| Platform | Native 4-bit matmul? | Format |
|---|---|---|
| NVIDIA datacenter | yes | MXFP4 / NVFP4 (E2M1) |
| AMD GPU | yes | MXFP4/MXFP6/MXFP8 |
| Intel GPU | yes (announced) | FP4 |
| Intel Xeon | **no** | AMX stops at 8-bit |
| AMD EPYC / ACE | **no** | MXFP8/MXINT8 native; MXFP4 convert-only |
| ARM (ISA) | **no** | MXFP6 in Armv9.7-A |
| Apple | yes (announced) | MXFP4, block-32 E8M0 |

---

## 3. What this implies for a CPU-first engine

1. **Keep the model format; make the compute mapping swappable.**
   Since MXFP4 is an OCP standard with broad hardware convergence, a serving engine should store
   weights exactly as the model ships them and map them onto whatever dot-product unit the CPU
   offers.
2. **4-bit storage, 8-bit compute is the endpoint, not a stopgap.**
   With no announced CPU 4-bit matmul, the stable design is: keep 4-bit weights, expand to 8-bit
   integers exactly, and accumulate in 32-bit — which is also what ARM's KleidiAI and ACE's
   MXFP4→MXFP8 conversion path do.
3. **Block-scale granularity matters more than it looks.**
   MXFP4 uses block-32 scales with power-of-two (E8M0) values; NVFP4 uses block-16 with FP8
   scales. Native CPU matrix support (ACE) is specified for **block-scale, E8M0** operands, so
   engines that coarsen activation scales to per-tensor or per-token lose that alignment.
4. **The real CPU levers are bandwidth and cache, not 4-bit instructions.**
   * *Memory bandwidth*: Xeon 6 introduced MCR/MRDIMM (~2× per module, 8800 MT/s);
     announced EPYC 9006 goes to 16 channels with MRDIMM at 12,800 MT/s.
     — [Intel MCR DIMM](https://www.intel.com/content/www/us/en/support/articles/000098737/processors/intel-xeon-processors.html)
     — [AMD EPYC 9006](https://www.amd.com/en/products/processors/server/epyc/9006-series.html)
   * *Per-CCD last-level cache*: 32 MB today, rising to 144 MB (announced EPYC "Venice-X") and
     320 MB per 64 cores (announced Xeon 7 Diamond Rapids).
     — [ComputerBase on EPYC 9006](https://www.computerbase.de/news/prozessoren/amd-epyc-9006-venice-details-zu-kernen-takt-x3d-cache-und-lp-in-vier-cpu-familien.98520/)
     — [TechPowerUp on Xeon 7](https://www.techpowerup.com/351893/intel-details-xeon-7-diamond-rapids-package-design-at-hot-chips)

   Cache capacity is the residency granularity for routed experts. Measured on a two-socket
   EPYC 9654 host: L3 is 32 MiB per 8-core CCD, and sequential read bandwidth at working sets
   within that CCD is roughly **2.7×** the DRAM figure (single-thread proxy measurement; the
   ratio is meaningful, the absolute values are not). Since a routed expert in MXFP4 occupies
   well under a megabyte, a 32 MiB CCD can hold on the order of tens of experts, and the
   announced 144–320 MB parts would hold several times more.

5. **GPUs: follow, do not lead.**
   For a project whose hosts are CPU-bound and whose GPU budget is fixed, the GPU roadmap
   matters mainly as *input format pressure* — i.e. new models arrive quantised in MXFP4 or
   NVFP4. The engine should be able to consume those formats; it need not chase GPU datatypes.

---

## 4. Design properties that capture the coming CPU generation

| Property | Why it matters | Status in this project |
|---|---|---|
| Runtime ISA dispatch (one build variant per instruction set) | new CPU instructions become an additive change | implemented (multiple ISA variants shipped) |
| Exact FP4→int8 weight expansion | needs no re-quantisation, so model accuracy is untouched | implemented |
| Block-scale (per-32) activations with E8M0 weights | matches how CPU matrix extensions are specified | by design |
| CCD/LLC-aware scheduling and expert-major blocking | converts larger caches into real speedup | planned |
| Non-temporal streaming for cold weights | keeps streaming traffic from evicting the hot set | planned |

---

## 5. Explicit non-claims

* No announced CPU provides native 4-bit matmul **as of the sources above**; a future
  architecture could change this. The design above is deliberately arranged so that such a
  change removes work (the expansion step) rather than invalidating the format.
* Capacities and dates for products marked *announced* are vendor statements, not shipping
  configurations; "bLLC" as a term appears only in consumer-segment leaks, not in Intel's Xeon
  material.
* The cache-bandwidth ratio quoted in §3.4 is our own single-thread measurement, not an
  official figure. No vendor publishes per-CCD L3 bandwidth.
* Claims about *how many models are shipped in FP4* are **not** made here; that would require a
  survey we have not run.
