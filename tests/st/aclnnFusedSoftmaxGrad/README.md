## 目录结构介绍

```
tests/st/aclnnFusedSoftmaxGrad/
├── cases.json
├── scripts/gen_data.py
├── src/main.cpp
└── run.sh

tests/ut/op_host/fused_softmax_grad/test_fused_softmax_grad_tiling.cpp
examples/aclnn/fused_softmax_grad/test_aclnn_fused_softmax_grad.cpp
```

本仓没有 ATK。`gen_data.py` 写 golden，`run.sh` 编译并按 `cases.json` 循环 launch。

## 运行

```bash
bash build.sh -u --st --ops=fused_softmax_grad
# 或：
cd tests/st/aclnnFusedSoftmaxGrad
bash run.sh
```
