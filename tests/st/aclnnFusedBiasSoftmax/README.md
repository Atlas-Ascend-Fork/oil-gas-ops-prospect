## 目录结构介绍

```
tests/st/aclnnFusedBiasSoftmax/
├── cases.json
├── scripts/gen_data.py
├── src/main.cpp
└── run.sh

tests/ut/op_host/fused_bias_softmax/test_fused_bias_softmax_tiling.cpp
examples/aclnn/fused_bias_softmax/test_aclnn_fused_bias_softmax.cpp
```

本仓没有 ATK。`gen_data.py` 写 golden，`run.sh` 编译并按 `cases.json` 循环 launch。

## 运行

```bash
bash build.sh -u --st --ops=fused_bias_softmax
# 或：
cd tests/st/aclnnFusedBiasSoftmax
bash run.sh
```
