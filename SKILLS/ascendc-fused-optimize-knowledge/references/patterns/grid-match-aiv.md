# grid-match-aiv

来源：cannbot tiling「核数 = min(任务数, AIV)」；Helix `grid-match-work` 的 AscendC 对应。

## 现象

`totalLine < 48` 仍 `blockDim=48`，空核也进 Pipeline。

## 做法

`usedAiv = min(aivNum, totalLine, ceil(totalLine / rowsPerIter))`。空核不启；小 payload 也不满 48 核，保证尽量 1 拍。不要回退成「永远 48 AIV」。
