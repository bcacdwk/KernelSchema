# F11 Supervisor验收修订

原判断把“不同通知流/额外full-empty API”直接当作不同核心设计的反例，证据不足。本轮收窄为`gap_kind: open_semantics`、`confidence: medium`，不再提出专用通知流字段。源码实际采用四独立PipelineAsyncUmma或共用MbarrierArray、以及省略producer acquire的事实全部保留。

复核了两个节点的完整MMA1/Recurrence依赖、single ScalarOutput顺序，以及固定源码`solution.py:154–164,349–387`。现有图已经给出：

1. `StateOperand(n)`覆盖Sbar之前等待上一S完成；`async_ordered_completion`把该完成传递到上一KS/QS及其操作数读取结束。
2. `read:U(n)`等当前U完成，再由同一serial role发布Ubar/UpdateOperand。因此对R槽的覆盖已经晚于其U读者。R下一轮及Udec复用另受上一S完成链保护。
3. single `prepare(c+5)`跟在`store:O(c).done`之后；该store等待O发布及Y完成，传递到U、Tinv、Inverse和Recurrence此前的scalar读取。
4. single把Ubar/Udec合并发布的区别已经由`publish:UpdateOperand`及MMA1的data边表示。一个物理通知存储是否被这些事件共享，是另一问题。

若full-empty协议在上述既有退役点归还slot，acquire只等待已经满足的条件。没有证据说明其必然添加无法由当前data/reuse/serial关系推出的等待或移动发布位置。barrier容量、API名称、附加指令数可能变化，但本审计没有量化这些差异的一阶成本，不能据此声称核心字段缺失。人为把release延后以制造新依赖，也不能证明full-empty形式本身要求那条新依赖。

对应标注修订：F11涉及的复合字段保留原主类别，其中单读S、合成成员或raw预取顺序仍是已确认的核心事实；root `realization.require[3].choice`只剩来源协议绑定与重复说明，其独立核心重要性改为`core: uncertain`。W/T/R已确认含核心文本由28组/34次声明降为27组/33次，另1组/1次未决。全算子相应由40组/48次降为39组/47次。158个文本组仍已全部审阅，主类别统计、17个findings、19条删改候选及字符节省均不变。
