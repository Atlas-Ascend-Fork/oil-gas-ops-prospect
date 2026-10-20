# AscendC 算子 pybind 封装

各 AscendC 算子的 **C++ 绑定** 与 **Python 对外接口**（autograd 接线、torch 回退、实现选择）。
绑定层直接调用 msopgen 生成的 `aclnn` 接口（如 `aclnnComplexMul`）。

测试用例统一在仓库根目录的 [`tests/`](../../tests/) 下。

## 目录结构

```
ascendc/pybind/
├── __init__.py              # 打包为 oil_gas_ops_prospect 包根
├── complex_mul.py           # oil_gas_ops_prospect.complex_mul
├── fused_bias_softmax.py    # oil_gas_ops_prospect.fused_bias_softmax
├── csrc/
│   ├── complex_mul/
│   │   └── bindings.cpp     # 直调 aclnnComplexMul / aclnnComplexMulGrad
│   └── fused_bias_softmax/
│       └── bindings.cpp     # 直调 aclnnFusedBiasSoftmax / aclnnFusedSoftmaxGrad
└── README.md
```

构建脚本在上一层，避免打进 Python 包：

- [`../pybind_extensions.py`](../pybind_extensions.py) — 扩展统一定义（根 `setup.py` 与就地编译复用）
- [`../pybind_setup.py`](../pybind_setup.py) — 就地编译入口

## 约定

`ascendc/pybind/csrc/<op_family>/` 下的 `*.cpp` 编译成扩展模块 **`<op_family>_npu`**。
Python API 放在 `ascendc/pybind/<op_family>.py`（与 `csrc/<op_family>/` 并存，互不冲突）。
新增算子：建 `csrc/<op_family>/`、写 `*.cpp` 与 `<op_family>.py`，两个构建入口都会自动发现 C++ 扩展。若扩展依赖多个 AscendC 算子，在 `pybind_extensions.py` 中登记完整依赖；根目录以 `--ops` 构建时，wheel 只打入依赖均已选择的扩展家族。未指定筛选的独立 wheel 构建和就地测试仍会构建全部扩展。

## 编译

正式形态是随 whl 一起构建（仓库根目录执行）：

```bash
bash build.sh --install    # 构建 OPP 包 → 安装 → 构建并安装 whl（含本扩展）
```

免装 whl 的就地编译，产物 `<op_family>_npu*.so` 落在本目录：

```bash
python3 ascendc/pybind_setup.py build_ext --inplace
```

`tests/ascendc/complex_mul/run_test.sh` 走的就是这条路径。

## 前置条件

- **OPP 包已安装到 CANN**，即 `opp/vendors/customize/op_api/lib/libcust_opapi.so` 存在。
- 环境内有 `torch` 与 `torch_npu`。
- 运行需要真实 NPU。

置 `OIL_GAS_OPS_SKIP_EXT=1` 可跳过扩展编译，只构建纯 Python 部分。
