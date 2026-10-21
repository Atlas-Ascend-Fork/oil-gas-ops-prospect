# Precision Acceptance Report Template

English | [简体中文](../../zh/develop/precision_acceptance_template.md)

A PR that adds or modifies an operator must include this report in its description. Copy the template below and fill in every item; delete the sections that do not apply and explain why.

For the precision standard see the [Accuracy Evaluation Standard for Open-Source Ecosystem Operators](https://gitcode.com/cann/opbase/blob/master/docs/en/ops_precision_standard/experimental_standard.md).

---

## 1. Basic Information

| Item | Content |
|------|---------|
| operator name / OpType | |
| backend | AscendC / Triton-Ascend / Composite-Torch |
| related Issue | #  |
| CANN version | |
| SoC model | |
| driver / firmware version | |
| `torch` / `torch_npu` version | |
| tester | |
| test date | |

## 2. Baseline Implementation

State what serves as the precision baseline and what device it runs on.

| Item | Content |
|------|---------|
| baseline implementation | e.g. an equivalent PyTorch expression |
| baseline device | e.g. CPU float64 / NPU float32 |
| comparison method | e.g. relative error `max(\|a-b\|) / max(\|b\|)` |

## 3. Precision Results

Fill in one row per dtype and shape combination. **The pass rate must be 100%.**

### Forward

| # | shape | dtype | format | Relative Error | Threshold | Result |
|---|-------|-------|--------|----------------|-----------|--------|
| 1 | | | | | | PASS / FAIL |
| 2 | | | | | | |

### Backward (if the operator supports autograd)

| # | shape | dtype | Output | Relative Error | Threshold | Result |
|---|-------|-------|--------|----------------|-----------|--------|
| 1 | | | | | | PASS / FAIL |

### Boundary and Exception Cases

| Case | Expected Behavior | Actual Behavior | Result |
|------|-------------------|-----------------|--------|
| empty tensor | | | |
| minimum shape | | | |
| unaligned shape | | | |
| non-contiguous tensor | | | |
| unsupported dtype / `norm` and other invalid arguments | raises an explicit exception | | |

## 4. Performance Results

Compared against the baseline implementation at the same shapes.

| # | shape | dtype | Baseline Latency (ms) | This Operator (ms) | Speedup | Notes |
|---|-------|-------|-----------------------|--------------------|---------|-------|
| 1 | | | | | | |

Measurement notes (warm-up count, iteration count, whether synchronized):

## 5. Device / Host Memory

| Item | Baseline | This Operator | Difference |
|------|----------|---------------|------------|
| peak device memory (MB) | | | |

## 6. Unit Tests

| Test Item | Command | Cases | Passed | Result |
|-----------|---------|-------|--------|--------|
| ophost tiling UT | `bash build.sh -u --ophost` | | | |
| ST | `bash build.sh -u --st --ops=<op>` | | | |
| precision / performance | `bash build.sh -u --precision --perf --ops=<op>` | | | |
| Triton CPU fallback precision | `python3 -m pytest tests/triton/real_fft -q` | | | |
| Triton AOT compile check | `python3 triton-ascend/tools/aot_compile_check.py` | | | |
| Composite-Torch CPU precision | `bash build.sh -u --composite` | | | |
| Composite-Torch NPU precision / performance | `python3 tests/composite/conv3d_custom/bench_conv3d_custom_perf.py --mode all` | | | |

Requirement: the full UT set passes and the logs contain no ERROR.

## 7. Open Issues

| # | Description | Severity | Workaround | Planned Closure |
|---|-------------|----------|------------|-----------------|
| 1 | | critical / major / minor | | |

> Critical and major issues **must be closed** before merging; minor issues need a workaround and a roadmap.

## 8. Conclusion

- [ ] precision meets the standard, pass rate 100%
- [ ] performance meets expectations
- [ ] the full UT set passes and the logs contain no ERROR
- [ ] open issues are closed, or have a workaround plus a roadmap

Overall conclusion: **PASS / FAIL**
