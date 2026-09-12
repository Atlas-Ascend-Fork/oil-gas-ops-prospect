# complex_mul / complex_mul_grad NPU 对比测试

`ComplexMul` 与 `ComplexMulGrad` 的精度与性能对比，基准为等价的 PyTorch 逐元素表达式。

## 算子语义

前向（复数逐点乘）：

```text
wr = ur * vr - ui * vi
wi = ur * vi + ui * vr
```

反向：

```text
gur =  gwr*vr + gwi*vi
gui = -gwr*vi + gwi*vr
gvr =  gwr*ur + gwi*ui
gvi = -gwr*ui + gwi*ur
```

全部输入/输出为 float32、`ND` 格式、同 shape。两个算子都是 **AIV-only**
（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`），按元素总数在 AIV 核间均分。

## 目录

```
tests/ascendc/complex_mul/
├── test_complex_mul_npu.py # 精度 / 性能 / autograd 封装对比
├── run_test.sh             # 安装 OPP 包 → 编译扩展 → 跑测试
└── README.md
```

被测的 pybind 封装（`complex_mul_npu` 扩展）是算子交付件，不在本目录，
见 [`ascendc/pybind/complex_mul.py`](../../../ascendc/pybind/complex_mul.py) 与
[`ascendc/pybind/csrc/complex_mul/`](../../../ascendc/pybind/csrc/complex_mul/)。

## 前置条件

1. 已构建出 OPP 包，产物在仓库根目录 `output/<SOC>/oil_gas_ops_prospect_opp_*.run`：

   ```bash
   cd /root/projects/oil-gas-ops-prospect/ascendc
   bash build.sh --pkg -v Ascend910B1
   ```

2. **测试需在带 NPU 的机器上运行**，且环境内有 `torch` 与 `torch_npu`。
   扩展编译可在无卡容器内完成，跑测试需真实 910B。

## 运行

```bash
cd tests/ascendc/complex_mul
bash run_test.sh

# 指定 shape
bash run_test.sh --shape 1 1 96 160 160

# 指定 OPP 包
OIL_GAS_OPS_RUN_PKG=/path/to/oil_gas_ops_prospect_opp_openEuler_aarch64.run bash run_test.sh
```

也可分步：

```bash
python3 test_complex_mul_npu.py --build-ext   # 编译扩展并测试
python3 test_complex_mul_npu.py               # 复用已编译扩展
```

## 输出

```text
[fwd] shape=(1, 1, 192, 320, 320) rel_err wr=... wi=...
[bwd] rel_err gur=... gui=... gvr=... gvi=...
[perf] fwd torch=...ms ascendc=...ms speedup=...x
[perf] bwd torch=...ms ascendc=...ms speedup=...x
[autograd] gur=... gui=... gvr=... gvi=...
PASS
```

`[autograd]` 一段走 `oil_gas_ops_prospect.complex_mul.complex_mul`，验证
`torch.autograd.Function` 封装的前反向接线。

退出码：`0` 精度达标；`1` 超阈值 / 无 NPU / 扩展缺失。

## 无卡环境下能做什么

跑不了本测试，但可以验证编译期正确性：

```bash
cd ascendc && bash build.sh --pkg          # 算子编译 + 打包
bash build.sh -u                           # 仓库根目录：全部无卡用例
```

以及确认 aclnn 接口已导出：

```bash
nm -D --defined-only \
  /usr/local/Ascend/cann-9.0.0/opp/vendors/customize/op_api/lib/libcust_opapi.so \
  | grep -i complexmul
```
