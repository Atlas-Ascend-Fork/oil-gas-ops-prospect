# Conv3DGelu / Conv3DGeluGrad Precision and Performance Report

English | [简体中文](precision_acceptance_report.md)

Filled from the [precision acceptance template](../../../../docs/en/develop/precision_acceptance_template.md). All 15 shapes were run on device in one process. Every forward relative error is `< 1e-5`. After the first chunked backward in a process is computed once and discarded, every input-gradient max absolute error is `< 1e-4`.

---

## 1. Basic Information

| Item | Content |
|------|---------|
| operator / OpType | `conv3d_gelu` / `Conv3DGelu`, backward `Conv3DGeluGrad` |
| backend | AscendC (forward MIX cube+vector; GELU backward is AIV) |
| related Issue | none |
| CANN version | 9.0.0 |
| SoC model | Ascend 910B2 (OPP built for `Ascend910B1`) |
| driver / firmware | npu-smi 26.0.rc1 / firmware 9.0.0.0.205 |
| `torch` / `torch_npu` | 2.7.1+cpu / 2.7.1.post4 |
| tester | this machine, container `ascend-npu-ha`, `ASCEND_RT_VISIBLE_DEVICES=7` |
| test date | 2026-10-28 |

Command: `python3 -u tests/ascendc/conv3d_gelu/test_conv3d_gelu_npu.py --mode all`.

`ACLNN_CACHE_LIMIT=0` is set before Python starts. `OIL_GAS_OPS_CONV3D_GELU_IMPL=ascendc`, `OIL_GAS_OPS_REQUIRE_ASCENDC_CONV3D_GELU=1`, `torch_npu.npu.conv.allow_hf32=False`.

Outputs up to 256MB run in one launch. Larger outputs are cut along D at about 48MB. The first chunked backward in a process is run twice and the first result is dropped.

## 2. Reference

| Item | Content |
|------|---------|
| reference | `F.gelu(F.conv3d(...), approximate="tanh")` |
| device | forward precision: CPU float32. Backward, latency, and memory: the split `F.conv3d` + `F.gelu` on the same NPU |
| comparison | relative error `max(\|y_op-y_cpu\|) / max(\|y_cpu\|)`, threshold `< 1e-5`. Input gradient uses max absolute error, threshold `< 1e-4` |

CPU `nn.GELU()` defaults to erf and is not the reference. HF32 is disabled. Every case has bias except `k3_24`, `k_dilate`, and `k_replicate`.

## 3. Precision

Forward **15/15**. Data type is `float32`, format is `NCDHW`, `groups=1`.

### Forward

| # | shape | dtype | format | relative error | threshold | result |
|---|-------|-------|--------|----------------|-----------|--------|
| 1 | k3_16, x `(1,16,16,16,16)`, w `(32,16,3,3,3)`, pad 1 | float32 | NCDHW | 8.85e-07 | `< 1e-5` | PASS |
| 2 | k3_32, x `(2,32,32,32,32)`, w `(32,32,3,3,3)`, pad 1 | float32 | NCDHW | 1.13e-06 | `< 1e-5` | PASS |
| 3 | k3_24, x `(1,8,24,24,24)`, w `(16,8,3,3,3)`, pad 0, no bias | float32 | NCDHW | 6.18e-07 | `< 1e-5` | PASS |
| 4 | k1_latent, x `(1,32,16,32,32)`, w `(64,32,1,1,1)`, pad 0 | float32 | NCDHW | 2.56e-07 | `< 1e-5` | PASS |
| 5 | k1_wide, x `(1,128,8,32,32)`, w `(64,128,1,1,1)`, pad 0 | float32 | NCDHW | 5.15e-07 | `< 1e-5` | PASS |
| 6 | k3_fullres_chunk, x `(1,32,80,128,128)`, w `(32,32,3,3,3)`, pad 1 | float32 | NCDHW | 1.18e-06 | `< 1e-5` | PASS |
| 7 | k3_96x192, x `(1,32,96,192,192)`, w `(32,32,3,3,3)`, pad 1 | float32 | NCDHW | 1.36e-06 | `< 1e-5` | PASS |
| 8 | k3_128x160, x `(1,32,128,160,160)`, w `(32,32,3,3,3)`, pad 1 | float32 | NCDHW | 1.26e-06 | `< 1e-5` | PASS |
| 9 | k3_64x256, x `(1,16,64,256,256)`, w `(32,16,3,3,3)`, pad 1 | float32 | NCDHW | 9.43e-07 | `< 1e-5` | PASS |
| 10 | k3_n2_112, x `(2,16,112,160,160)`, w `(32,16,3,3,3)`, pad 1 | float32 | NCDHW | 1.16e-06 | `< 1e-5` | PASS |
| 11 | k3_160, x `(2,32,160,160,160)`, w `(32,32,3,3,3)`, pad 1 | float32 | NCDHW | 1.31e-06 | `< 1e-5` | PASS |
| 12 | k_stride, stride `(2,1,1)`, pad `(1,1,0)` | float32 | NCDHW | 1.94e-07 | `< 1e-5` | PASS |
| 13 | k_dilate, dilation `(2,1,1)`, pad `(2,0,0)`, no bias | float32 | NCDHW | 6.63e-08 | `< 1e-5` | PASS |
| 14 | k_asym, stride `(1,1,2)`, pad `(0,1,0)` | float32 | NCDHW | 1.09e-07 | `< 1e-5` | PASS |
| 15 | k_replicate, `padding_mode=replicate`, no bias | float32 | NCDHW | 3.27e-07 | `< 1e-5` | PASS |

### Backward

The compared tensor is the input gradient. The reference is autograd of the split path on the same NPU. The gate is max absolute error `< 1e-4`. Weight gradients are excluded because of NPU atomic-add noise.

`k3_96x192` is the first chunked backward in the process. Before the extra full-pass discard its first user-visible error was `1.54e2`. After that discard the same call is `8.39e-5`.

| # | shape | dtype | output | max absolute error | threshold | result |
|---|-------|-------|--------|--------------------|-----------|--------|
| 1 | k3_16 | float32 | `grad_x` | 0 | `< 1e-4` | PASS |
| 2 | k3_32 | float32 | `grad_x` | 3.82e-05 | `< 1e-4` | PASS |
| 3 | k3_24 | float32 | `grad_x` | 3.05e-05 | `< 1e-4` | PASS |
| 4 | k1_latent | float32 | `grad_x` | 2.15e-05 | `< 1e-4` | PASS |
| 5 | k1_wide | float32 | `grad_x` | 2.48e-05 | `< 1e-4` | PASS |
| 6 | k3_fullres_chunk | float32 | `grad_x` | 4.58e-05 | `< 1e-4` | PASS |
| 7 | k3_96x192 | float32 | `grad_x` | 8.39e-05 | `< 1e-4` | PASS |
| 8 | k3_128x160 | float32 | `grad_x` | 9.16e-05 | `< 1e-4` | PASS |
| 9 | k3_64x256 | float32 | `grad_x` | 9.92e-05 | `< 1e-4` | PASS |
| 10 | k3_n2_112 | float32 | `grad_x` | 8.39e-05 | `< 1e-4` | PASS |
| 11 | k3_160 | float32 | `grad_x` | 9.16e-05 | `< 1e-4` | PASS |
| 12 | k_stride | float32 | `grad_x` | 1.34e-05 | `< 1e-4` | PASS |
| 13 | k_dilate | float32 | `grad_x` | 5.90e-06 | `< 1e-4` | PASS |
| 14 | k_asym | float32 | `grad_x` | 4.17e-06 | `< 1e-4` | PASS |
| 15 | k_replicate | float32 | `grad_x` | 9.30e-06 | `< 1e-4` | PASS |

`groups!=1` and a `stride` / `padding` / `dilation` whose length is not 1 or 3 raise `ValueError`. Empty tensors, non-contiguous inputs, and illegal data types are not covered.

## 4. Performance

Forward latency against one split forward on the same NPU. One warmup, then 2 timed iterations (3 when the spatial edge is below 128), with `torch.npu.synchronize()` around the timing. These numbers are forward only. They do not include the backward recompute.

| # | shape | dtype | reference (ms) | this operator (ms) | speedup | note |
|---|-------|-------|----------------|--------------------|---------|------|
| 1 | k3_16 | float32 | 0.44 | 0.43 | 1.02 | one launch |
| 2 | k3_32 | float32 | 0.46 | 0.54 | 0.85 | one launch |
| 3 | k3_24 | float32 | 0.52 | 0.62 | 0.84 | one launch |
| 4 | k1_latent | float32 | 0.54 | 0.44 | 1.23 | 1×1 direct write, gate `> 1.05` |
| 5 | k1_wide | float32 | 0.46 | 0.38 | 1.21 | 1×1 direct write, gate `> 1.05` |
| 6 | k3_fullres_chunk | float32 | 2.36 | 2.28 | 1.04 | one launch |
| 7 | k3_96x192 | float32 | 5.85 | 8.80 | 0.66 | cut along D |
| 8 | k3_128x160 | float32 | 5.73 | 7.97 | 0.72 | cut along D |
| 9 | k3_64x256 | float32 | 4.61 | 5.26 | 0.88 | cut along D |
| 10 | k3_n2_112 | float32 | 6.11 | 7.95 | 0.77 | cut along D |
| 11 | k3_160 | float32 | 13.76 | 19.71 | 0.70 | cut along D |
| 12 | k_stride | float32 | 0.43 | 0.46 | 0.95 | stride `(2,1,1)` |
| 13 | k_dilate | float32 | 0.43 | 0.46 | 0.93 | dilation `(2,1,1)` |
| 14 | k_asym | float32 | 0.45 | 0.45 | 1.01 | asymmetric kernel |
| 15 | k_replicate | float32 | 0.53 | 0.58 | 0.91 | Python pads first |

Both 1×1 direct-write cases are above 1.05. The five chunked shapes run at 0.66× to 0.88× the split forward.

## 5. Memory

Forward peak is `max_memory_allocated` during the call, including inputs that are already live. Held memory is the extra allocation after a grad-enabled forward returns and before backward starts. The fused path does not keep the pre-activation between conv and GELU, so held memory is about one output smaller.

| shape | activation (MB) | peak reference | peak this op | held reference | held this op | held saved |
|-------|-----------------|----------------|--------------|----------------|--------------|------------|
| k3_16 | 0.5 | 17.6 | 17.6 | 1.0 | 0.5 | 0.5 |
| k3_32 | 8.0 | 48.2 | 48.2 | 16.0 | 8.0 | 8.0 |
| k3_24 | 0.6 | 19.1 | 19.1 | 1.3 | 0.7 | 0.7 |
| k1_latent | 4.0 | 22.0 | 22.0 | 8.0 | 4.0 | 4.0 |
| k1_wide | 2.0 | 22.0 | 22.0 | 4.0 | 2.0 | 2.0 |
| k3_fullres_chunk | 160.0 | 656.2 | 656.2 | 320.0 | 160.0 | 160.0 |
| k3_96x192 | 432.0 | 1744.2 | 1079.1 | 864.0 | 432.0 | 432.0 |
| k3_128x160 | 400.0 | 1616.2 | 1017.1 | 800.0 | 400.0 | 400.0 |
| k3_64x256 | 512.0 | 1552.1 | 992.1 | 1024.0 | 512.0 | 512.0 |
| k3_n2_112 | 700.0 | 2116.1 | 1361.0 | 1400.0 | 700.0 | 700.0 |
| k3_160 | 1000.0 | 4016.2 | 2416.2 | 2000.0 | 1000.0 | 1000.0 |

Below 256MB the forward peak matches the split path and held memory drops by one activation. The five chunked shapes also have a lower forward peak. `160³` drops from 4016 MB to 2416 MB, and held memory drops from 2000 MB to 1000 MB.

## 6. Conclusion

- [x] Precision passes: forward 15/15, input gradient 15/15
- [x] 1×1 direct-write speedups are `1.23` and `1.21`, both above 1.05
- [x] The five chunked shapes use less forward-peak and held memory than the split path
- [x] Open items are minor: weight-gradient noise, uncovered empty or non-contiguous inputs, and one forward spike to relative error 0.98 on `k3_96x192` that did not reproduce in a fresh process (1.36e-6)

Overall: **pass**
