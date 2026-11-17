# PageIndex 覆盖顺序复核

固定 v55 `gqa_decode_paged.py`：

- `1224–1246`：K 迭代 s 开头预取 PageIndex(s+1)，commit 后以 wait_group(1) 取本轮。
- `1249–1269`：之后才 await Vturn；在最后 K slice acquire 后 signal Kturn，且 signal 先于最后 TMA copy。
- `1315–1321`：V 启动先消费两次 Kturn，逐次释放 Kturn 的对向许可。
- `1343–1353`：V 迭代 i 再 await Kturn，然后 capture PageIndex(i)，最后一次 V acquire 后才释放 Vturn。

当 N≥5，源码允许如下事件偏序：K0 signal → V ramp0 → K1 signal → V ramp1 → K2 signal；此时 V0 可以捕获 PageIndex0，但 K2 也可以结束自身发射并进入 K3。K3 在 await V0 turn 之前就发起 PageIndex4 的 cp.async 写入槽0。没有找到 PageIndex4 写入与 V0 capture 之间的显式同步顺序。wait_group(1) 是 K warp 的复制完成等待，不能强制另一个 warp 先读；pt_stages>Lead+1 的容量断言也不单独提供该顺序。

因此本轮不维持 SELECTION 的“该复用已由轮次保证”结论。保留真实四槽及预取位置，删去曾临时添加但源码不支持的 `nextpages(i+3) <- V.pagecopy(i)` 保护边。记录为源实现未决的 PageIndex 读写竞态边界；没有 GPU 验证，也未改 kernel。需要新 reviewer 检查是否存在未识别的原语保证，不能为使图成立补同步。

## 独立终审复核

2026-11-17，新 reviewer 再读上述循环及 KV producer/consumer advance，结论仍为未决。K2 的四个 KV 槽重用 K0 的已释放槽；这些 acquire 只约束 K0 operand 消费，不要求 V0 已读取 PageIndex0。K2 的最后 acquire 后便 signal Kturn，K3 接着发起 Page4 预取；V0 虽已获得开始读取的许可，尚无必须先于该复制发生的源顺序。Q1 的更深 KV 环也不产生这个缺失的 PageIndex 读者交接。

四份 YAML 的 nextpages 位置与真实源码一致，未添加 `V.pagecopy -> nextpages` 边。有限图是已声明动作的依赖检查，并不检查共享地址读写冲突；T_b70 或 N5 检查无环不消除这里的风险。SELECTION 的旧安全断言已同步更正。
