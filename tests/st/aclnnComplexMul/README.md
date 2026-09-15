## 目录结构介绍

```
tests/st/aclnnComplexMul/          # 两段式 aclnnComplexMul + numpy 真值
├── cases.json
├── scripts/gen_data.py
├── src/main.cpp
└── run.sh

tests/ut/op_host/complex_mul/test_complex_mul_tiling.cpp
examples/aclnn/complex_mul/test_aclnn_complex_mul.cpp
```

本仓没有 ATK。`gen_data.py` 写 golden，`run.sh` 编译并按 `cases.json` 循环 launch。

## 运行

```bash
bash build.sh -u --st --ops=complex_mul
# 或：
cd tests/st/aclnnComplexMul
bash run.sh
```
