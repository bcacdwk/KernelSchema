# Extract 文字试读任务

为后续作者准备一份有限的 NVFP4 MoE 机制说明。使用真实源码，按正式提取方法自行安排阅读、判断和交接；不运行 Runner、检查器、GPU 或正式提取，不修改算子 YAML。只讨论下面明确列出的路径，结论不代表全算子 P2 或其他阶段获批。

## 本次材料与范围

算子入口位于 `kernel_wiki/aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/`。以 `problem/` 的任务材料、`candidate/solution.py` 的调用选择和 `candidate/nvfp4_moe.cu` 的实际实现为事实依据。本次聚焦正常 NVFP4 packed 计算链、cfg98/cfg99 的实际路径及同一计算链可用的 small prepare；不要求研究全部显式配置、其他 FC2 族、ZERO2 或完整图服务。需要判断当前路径的关系时可继续读被调用函数。

已有旧稿可用来定位值与工作，字段和选型结论都须按现行规范及源码核对：

```text
.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/root.yaml
.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/split_store.yaml
.schema_extract/v2/runs/feb0c655-4e13-43ac-817d-1f457f31c029/checkpoints/92874cf7540ccce5cdc4045c62f400aa4156ab1168214ff088336507375f77fb/artifacts/fused_small.yaml
```

先读 `EXTRACT_SCHEMA.md`，根据其导航读 P2、P3、P4、P5 与 REVIEW/common。材料只用于限域角色推演，没有新的正式 P1 批准；必要的 Contract 事实按给定源码核实。未迁移工具及正式阶段结论的边界见 `schema_extract/EXECUTOR_ALIGNMENT.md`。

## 停点与反馈

留下能够交给下一位的短说明：实际采用的计算与输出路径，有价值的差异及记录去处，当前能选择什么、还缺什么依据；接下来哪些内容需要写到数学、调度或流水线，哪些文件由哪个阶段处理。依据足够便停止，不补全所有动作或穷举配对。

用未来 Worker 的身份记下实际阅读与判断，再从接手 Reviewer 的角度说明会核对什么、哪些问题会退到哪里；这只是单个代理的角色推演，不是独立审批。没有问题就照实说明，不要求寻找错误。区分文案不清、源码尚未核实、输入缺项、旧稿未迁移及正常未完成内容。

不阅读本目录第一轮评审及另一位试读者的记录。请提出有助于正常推进工作的建议，不以增加条目或推翻既定设计为目标。
