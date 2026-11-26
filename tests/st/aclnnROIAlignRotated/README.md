## 目录结构介绍

```text
tests/st/aclnnROIAlignRotated/     # 两段式 aclnnROIAlignRotated + 纯 torch 真值
├── cases.json                     # 输入 shape、K、PH/PW、sampling_ratio、scale、angles
├── scripts/gen_data.py            # 生成 input/rois/attrs 与 golden
├── src/main.cpp                   # 两段式 aclnn 调用 + 结果落盘
├── src/CMakeLists.txt
├── run.sh
└── README.md
```

`gen_data.py` 调用仓内纯 torch 参考实现 `torch_roi_align_rotated` 生成 golden，
边界语义与 detectron2 `ROIAlignRotated_cuda.cu` 一致。

## 运行

```bash
# 仓库根目录
bash build.sh -u --st --ops=roi_align_rotated

# 或：
cd tests/st/aclnnROIAlignRotated
bash run.sh
```

## 用例

- `detectron2_5x5_4x4`：detectron2 官方单测的最小形态；
- `multi_roi_3ch_5x5_sr2`：多 ROI + `sampling_ratio=2` + 多角度；
- `multi_batch_scale_0.5`：多 batch + `spatial_scale=0.5` + 多角度。

退出码：`0` 比对通过；`1` 失败。
