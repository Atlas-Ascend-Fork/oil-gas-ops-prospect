# 安全声明

## 运行用户建议

基于安全性角度考虑，不建议使用 root 等管理员类型账户执行任何命令，遵循权限最小化原则。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统 umask 值为 0027 及以上，保障新增文件夹默认最高权限为 750，新增文件默认最高权限为 640。
- 建议用户对个人隐私数据、商业资产、源文件和算子开发过程中保存的各类文件等敏感内容做好权限控制等安全措施。本项目面向油气勘探场景，**地震数据体、测井数据、速度模型等业务数据通常属于高敏感资产**，请对输入数据文件、中间道集与成像结果目录做好权限管控，设定的权限建议参考[A-文件（夹）各场景权限管控推荐最大值](#a-文件夹各场景权限管控推荐最大值)。
- 算子运行时可能会缓存算子编译文件，存储在运行目录下的 `kernel_meta_*` 文件夹内，加快后续算子的调用速度，用户可根据需要自行对生成后的相关文件进行权限控制。
- 用户安装和使用过程需要做好权限控制，建议参考[A-文件（夹）各场景权限管控推荐最大值](#a-文件夹各场景权限管控推荐最大值)进行设置。

## 构建安全声明

- 在源码编译安装本项目时，需要您自行编译，编译过程中会生成一些中间文件，建议您在编译完成后对中间文件做好权限控制，以保证文件安全。本项目的编译中间件与产物集中在以下位置，便于统一管控：
  - `ascendc/_work/`：msopgen 生成的临时算子工程与 ophost UT 构建目录
  - `output/`：OPP `.run` 算子包与 Python whl 产物
  - `build/`、`*.egg-info/`：Python 扩展编译中间件
- 本项目的 C++ 目标统一应用了安全编译选项（`-fstack-protector-strong`、`-D_FORTIFY_SOURCE=2`、`-fPIC`、`-Wl,-z,relro`、`-Wl,-z,now`、`-Wl,-z,noexecstack`），实现见 [`cmake/security_flags.cmake`](cmake/security_flags.cmake)。
- 安装 OPP `.run` 算子包会向 CANN 安装目录的 `opp/vendors/customize/` 写入文件，请确认执行账户对该目录具备必要且最小的写权限。

## 运行安全声明

- 建议用户结合运行环境资源状况编写对应算子调用脚本。若算子调用脚本与资源状况不匹配，如生成输入数据或标杆计算结果使用空间超出内存容量限制、脚本在本地保存数据超过磁盘空间大小等情况，可能会引发错误并导致进程意外退出。勘探场景的三维数据体通常较大，请特别关注显存与内存容量。
- 算子在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看日志文件等方式。
- 本项目算子通过 [PyTorch](https://gitee.com/ascend/pytorch) 方式调用时，可能会因为版本不匹配导致运行错误，具体请参考 [PyTorch 安全声明](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。
- 本项目通过 `OIL_GAS_OPS_` 前缀的环境变量控制产物路径与源码/whl 切换（见 README.md）。这些环境变量会影响文件读写位置与实际加载的代码，请勿在不可信环境中由外部输入直接设置。

## 公网地址声明

本项目**源码与配置文件**中包含的公网地址声明如下所示（Markdown 文档中用于跳转的社区链接不在此列）：

| 类型 | 开源代码地址 | 文件名 | 公网IP地址/公网URL地址/域名/邮箱地址/压缩文件地址 | 用途说明 |
| :--: | :--: | :-- | :-- | :-- |
| 依赖 | 不涉及 | cmake/third_party/googletest.cmake | [https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) | 从 gitcode 三方件镜像下载 googletest 源码，作为 ophost 单元测试的编译依赖。仅在环境中找不到已安装的 gtest 时触发下载 |
| 三方件声明 | 不涉及 | Third_Party_Open_Source_Software_List.yaml<br>Third_Party_Open_Source_Software_Notice | [https://gitcode.com/cann-src-third-party/googletest](https://gitcode.com/cann-src-third-party/googletest) | googletest 上游获取地址，用于三方件清单声明，不在构建期访问 |
| 三方件声明 | 不涉及 | Third_Party_Open_Source_Software_List.yaml<br>Third_Party_Open_Source_Software_Notice | [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) | PyTorch 上游获取地址，用于三方件清单声明。PyTorch 由用户自行安装，本仓不下载、不分发 |
| 三方件声明 | 不涉及 | Third_Party_Open_Source_Software_List.yaml<br>Third_Party_Open_Source_Software_Notice | [https://gitee.com/ascend/pytorch](https://gitee.com/ascend/pytorch) | torch_npu 上游获取地址，用于三方件清单声明。由用户自行安装，本仓不下载、不分发 |
| 三方件声明 | 不涉及 | Third_Party_Open_Source_Software_List.yaml<br>Third_Party_Open_Source_Software_Notice | [https://gitcode.com/cann/triton-ascend](https://gitcode.com/cann/triton-ascend) | Triton-Ascend 上游获取地址，用于三方件清单声明。由用户自行安装，本仓不下载、不分发 |

---

## 漏洞机制说明

[漏洞管理](https://gitcode.com/cann/community/blob/master/security/security.md)

## 附录

### A-文件（夹）各场景权限管控推荐最大值

| 类型 | Linux权限参考最大值 |
| -------------- | --------------- |
| 用户主目录 | 750（rwxr-x---） |
| 程序文件（含脚本文件、库文件等） | 550（r-xr-x---） |
| 程序文件目录 | 550（r-xr-x---） |
| 配置文件 | 640（rw-r-----） |
| 配置文件目录 | 750（rwxr-x---） |
| 日志文件（记录完毕或者已经归档） | 440（r--r-----） |
| 日志文件（正在记录） | 640（rw-r-----） |
| 日志文件目录 | 750（rwxr-x---） |
| Debug文件 | 640（rw-r-----） |
| Debug文件目录 | 750（rwxr-x---） |
| 临时文件目录 | 750（rwxr-x---） |
| 维护升级文件目录 | 770（rwxrwx---） |
| 业务数据文件 | 640（rw-r-----） |
| 业务数据文件目录 | 750（rwxr-x---） |
| 密钥组件、私钥、证书、密文文件目录 | 700（rwx------） |
| 密钥组件、私钥、证书、加密密文 | 600（rw-------） |
| 加解密接口、加解密脚本 | 500（r-x------） |
