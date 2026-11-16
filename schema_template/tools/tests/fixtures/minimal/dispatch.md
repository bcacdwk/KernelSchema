# scalar_copy

## 范围

只用于工具检查的合成标量复制任务；资料引用使用 testing.invalid，不对应真实设备或实现

## 完整设计

| 设计 | 具体机制与保留理由 | 关系 |
| --- | --- | --- |
| [root](root.yaml) | 一个 worker 内由单个串行 role 复制标量，提供最小结构正例 | 阅读入口，保存唯一 Contract |
