# host-matchargs-off

## 现象

CANN 9 `NnopbaseMatchArgs` 按 shape 复用旧 executor，原地写回会空跑。

## 做法

`op_host/extra.cmake` 去掉融合算子 autogen 里的 MatchArgs。保持关闭。

autogen MatchArgs 保持关闭。同 shape 的二次 `GetWorkspaceSize` 在 pybind 侧用自管 executor 缓存跳过，但每次必须 Bind + launch；禁止走「命中即 SUCCESS」空跑。
