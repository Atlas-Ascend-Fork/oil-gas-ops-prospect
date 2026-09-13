# 评测

在 **已挂载本仓库** 的 NPU 容器里执行。容器入口由环境决定，不要把机器地址写进提交物。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
export ASCEND_RT_VISIBLE_DEVICES=0
export ACLNN_CACHE_LIMIT=0
cd /root/work/workspace/mcg/oil-gas-ops-prospect
```

`docker exec` 用 `/bin/bash -c`，不要 `-lc`。

## 正确性

```bash
# 无卡 / CPU 回退
python3 -m pytest tests/triton/real_fft/test_triton_fft_real.py -q

# 有卡：同一套 pytest（NPU 用例自动启用）
python3 -m pytest tests/triton/real_fft/test_triton_fft_real.py -q
```

可选官方三路对拍（CPU `torch.fft` / NPU `torch.fft` / fused）：

```bash
python3 tests/triton/real_fft/verify_triton_fft_real_128_3d.py \
  --shape 1 128 128 128 1 --axes 1 --rounds 1
```

## 性能（相对同设备 torch.fft）

```bash
python3 /root/work/tmp/oil-gas-qa/bench_fft.py
```

必须同时看三类，禁止只报 geomean：

1. 常规小/中末轴（`[8,256]`、`[4,64,64]`、`[2,128,128]`、`128³` 末轴）
2. 非末轴（`32³` / `128³` dim=1）
3. 业务 5D `[1,35,128,128,128]` 的 W/H/D

## 编译

```bash
python3 triton-ascend/tools/aot_compile_check.py
```

无 CANN 的开发机可跳过，但 NPU 容器上改 kernel 后应跑。

## 通过标准

- 精度：相对 `torch.fft` 相对误差 ≤ `3e-3`（官方用例更严处从其规定）
- 性能：相对本轮开始前的 baseline 表逐行对比；5D 各轴加速比不得明显回退
- 不改官方 PASS 标准来「过线」
