# row_softmax_fp32_v1

## 范围

手工编写并审阅的 R0 材料，覆盖 Contract 的四个 shape：M∈{1,32}、N∈{128,1024}。
单节点 root 没有额外 when 限制，不能推导为任意 N、任意 dtype 或非连续布局。
运行设备、架构与栈版本尚未冻结；root 的 Hardware 区如实标记这些缺项，
结构检查不代表 DOC2 的具体设备资料或执行准入已经完成。
H20 只是在环境资格阶段首先检查的候选，不是本材料声称已选定的设备。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 一 Triton program/CTA 独占整行；FP32 max-subtract、exp、sum、div；一次读取及写回，无显式全局中间量或跨 CTA 协作 | 阅读入口，保存唯一共享 Contract；手工 R0，非模型 design 结果 |

`num_warps` 的 4/8 选择、FP32 归约树及编译调度是保持该设计的 coder 自由度，
进入实际源码与证据，不增加设计 revision。多行 program、全局物化或工作归属
改变是设计变化，不能以更快或数值通过为由算作 R0 的实现细节。
二幂 N 不引入额外尾 tile 义务；任意合法 storage_offset 的 ABI 义务仍在。
program 内部值的驻留不证明实际无 spill；需要后续编译/运行证据。

Design 的输入不包含此 root、此目录或手审义务表。Optimize 只读取请求显式
选中的当前完整 Schema；若为 Rd/R′，按其实际义务审查，不能套用 R0 常量。
官方教程允许作为公开参考；本例说明借助参考的设计/实现，不声称纯 Schema
重建或模型未见过公开教程。
