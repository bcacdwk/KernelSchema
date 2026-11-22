# FP32 vector_add

## 范围

合法长度是 1..65536；输入、输出和判据由 root 的共享 Contract 固定。root 适用 1..1024，tiled 适用 513..65536。重叠区 513..1024 明确选择 root；1025..65536 选择 tiled。Model.forward 必须根据长度显式选路，root 没有隐式兜底。

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 一CTA、1024元素的短向量方案，减少启动内分工 | 短输入完整设计；重叠区优先 |
| [tiled](tiled.yaml) | 每CTA 256元素，多CTA覆盖长输入并掩码尾块 | 完整长输入设计；数学区与局部执行协议实际from root，调度区完整本地定义 |

当前无既有成绩。固定七个案例覆盖边界与尾块，但不缩小合法域。借用区域在 tiled 当前调度上下文绑定；来源 when 和 REALIZATION 不随 from 传播，四warp要求在两节点各自记录。

root Version 2 补充已知软件身份，来源固定为 `KernelSchema@51f6b2399d29bafbeacf401de45afa7011160dc8:schema_workflow/examples/vector_add/environment.json`。记录来自 Agate SDK `get_env('L20D')` 的服务缓存，证明该缓存报告的设备与版本；不是本次候选实测或未来分配保证。它不提供完整硬件/编程语义，Wiki 未绑定入口仍是可见缺项。tiled 文件未改，其有效 Contract 随 root 更新；算法、选路与 Golden_reference 不变。
