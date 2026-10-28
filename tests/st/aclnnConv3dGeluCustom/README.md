# aclnnConv3dGeluCustom 系统测试

## 目录结构介绍

```text
tests/st/aclnnConv3dGeluCustom/
├── cases.json
├── scripts/gen_data.py
├── src/main.cpp
└── run.sh

tests/ut/op_host/conv3d_gelu/test_conv3d_gelu_tiling.cpp
examples/aclnn/conv3d_gelu/test_aclnn_conv3d_gelu.cpp
```

`gen_data.py` 写 CPU float32 `gelu_tanh(conv3d)` golden。`run.sh` 编译并按 `cases.json` 循环 launch。

## 运行

```bash
bash build.sh -u --st --ops=conv3d_gelu
bash build.sh -u --ophost
```
