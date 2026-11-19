# 四算子直接审修与正式目录清理

本轮任务已完成：四个新的 xhigh 分两轮完成 GEMM、MoE、MLA、GDN 的直接 Schema 审修；随后完成正式目录统一清理，并由另一位新代理独立复核通过。

## 当前成果

六算子共15份现行YAML；当前文件、版本、精简元数据与来源入口见 [CURRENT_SCHEMA_INDEX.json](CURRENT_SCHEMA_INDEX.json) 和 [Schema Wiki](../../schema_wiki/README.md)。

| 本轮审修算子 | 当前版本 | 结果 |
| --- | --- | --- |
| GEMM | root 5、wide 2 | [审修报告](reviews/gemm/REVIEW.md)：修正寄存器副本账、TMEM归还与尾批计算时序、跨工作项事件身份等；目录及有限事件检查通过。 |
| MoE | root 5、split_k 2、mega 3 | [审修报告](reviews/moe/REVIEW.md)：修正舍入顺序、padding访存、ACT加载/存储账、配置与验收描述；目录及CPU核对通过。 |
| MLA | root 6 | [审修报告](reviews/mla/REVIEW.md)：修正数值链、存储与最终退出依赖，撤销tiny路径伪缺陷判断；目录、事件与CPU核对通过。 |
| GDN | root 3、segmented 3 | [审修报告](reviews/gdn/REVIEW.md)：修正BF16舍入、分段近似、传播状态、区域与流水线；目录、事件与CPU核对通过。 |

当前正文保留有依据的来源不足和源实现限制，不以删历史为由去掉这些事实。没有重跑extract、P6或GPU验证，没有修改kernel源码。

## 清理结果

- 正式交付目录只保留现行YAML、dispatch与每算子的精简`.extraction/current.json`；过程叙事、旧回执、完整审修记录和失败检查已移出。
- 856个历史文件迁移归档，另保全2份清理前已删除文件的Git历史原件；85个有源码可再生的字节码缓存已删除，唯一历史内容没有销毁。
- 明示历史目录归入`schema_discussion/archive/`，旧路径、当前路径及SHA见[位置映射](cleanup/LOCATIONS.json)。保留41份正式issue决定台账及Git忽略的运行证据，维持其既有用途和接口。
- 15份审修后YAML在清理中保持原字节；六目录最终检查、11项prompt/freeze/input离线回归通过；活动代码与运行态基线保持。
- [独立最终复核](cleanup/INDEPENDENT_REVIEW.md)通过，624个活跃本地链接有效，无待修清理项。最后发现的一句Sparse过程措辞已改为当前证据覆盖说明，真实L20D来源记录保留。

完整清理结果见 [REPORT.md](cleanup/REPORT.md) 和 [verification.json](cleanup/verification.json)。最初残留情况保留在[清理前记录](FORMAL_DIRECTORY_HISTORY.md)。本轮未提交或推送。
