## 目录结构介绍

```text
tests/st/aclnnROIAlignRotatedGrad/  # 两段式 aclnnROIAlignRotatedGrad + 纯 torch 真值
├── cases.json                      # 输入 shape、K、PH/PW、sampling_ratio、scale、angles
├── scripts/gen_data.py             # 生成 grad_output/rois/attrs 与 golden grad_input
├── src/main.cpp                    # 两段式 aclnn 调用 + 结果落盘
├── src/CMakeLists.txt
├── run.sh
└── README.md
```

`gen_data.py` 对仓内纯 torch 参考实现 `torch_roi_align_rotated` 求导生成 golden，
反向散射累加语义与 detectron2 `ROIAlignRotated_cuda.cu` 一致。

## 运行

```bash
# 仓库根目录
bash build.sh -u --st --ops=roi_align_rotated

# 或：
cd tests/st/aclnnROIAlignRotatedGrad
bash run.sh
```

退出码：`0` 比对通过；`1` 失败。
