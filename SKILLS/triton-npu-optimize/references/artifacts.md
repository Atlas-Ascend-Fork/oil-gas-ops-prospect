# 产物

| 路径 | 是否提交 | 内容 |
|---|---|---|
| `triton-ascend/real_fft/triton_fft_real.py` | 是 | 唯一默认改码点 |
| `tests/triton/real_fft/` | 仅当接口/分支必须测 | 官方用例 |
| `SKILLS/` | 是 | 本技能包 |
| `extra-info/fft-optimize/baseline.md` | 否 | Stage 0 数字 |
| `extra-info/fft-optimize/opt-note.md` | 否 | 轮次账本 |
| `extra-info/fft-optimize/opt-round-N/` | 否 | 可选：该轮 diff 说明、bench 原文 |

`opt-note.md` 最低结构：

```markdown
# FFT optimize note

## Baseline
（贴三类加速比）

## Round 1
- 方向：<pattern 名>
- 改动：<一句话>
- 结果：常规末轴 / 非末轴 / 5D
- 结论：keep / revert
```
