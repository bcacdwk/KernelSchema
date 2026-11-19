# 正式目录残留过程内容：待统一收口

用户要求已记录：先由四个新的 xhigh 分两轮，分别直接检查并修订 GEMM、MoE、MLA、GDN 的既有 Schema；全部结束后，再由新的子代理统一检查正式目录、迁移或删除历史内容。FMHA/Sparse 的专项审修已经完成。本轮不重新运行 extract。

此前只读核对的事实：当前六算子目录含15份 YAML、6份 dispatch、6份原 receipt 和4份人工审修记录。没有独立旧稿、before/after备份或 rejected/response/fail 原始日志混入这些正文目录；但正式入口、dispatch和元数据并非完全干净。

## 已发现的残留

- schema_wiki/README.md 有批次日期、xhigh角色安排、未重跑extract/P6、目录搬迁等过程说明。
- FMHA dispatch.md 含P2/P4交办、本轮同一worker/reviewer、旧invalid_receipt/P6/published:false等过程叙述。
- Sparse dispatch.md 含人工审修日期、前后字节、模型角色、未新增P6等叙述。同段源控制区布局缺陷属于有效语义限制，必须保留。
- 四份人工审修JSON混有before/after、修订过程和角色安排；Sparse记录含中途exit_code=1的检查，FMHA记录含格式/预算检查过程。
- MLA/GEMM人工记录的approval_scope仍保留等待后续精简/复核的旧阶段说法；旧FMHA/Sparse receipt有pending_user_decision等历史状态。历史事实不能改写成原P6批准当前新字节。
- schema_extract/issues/README.md有具体批次收口叙事；issues/archive/README.md有专项审计尚未完成的旧说法。

## 当前正文以外的历史区

schema_wiki/archive、schema_extract/archive、schema_docs/archived 保存旧结果、设计、评估或讨论。schema_extract/issues/archive 是正式问题与决定台账，须保留其工具接口和稳定ID；其中过程性Markdown可以另行归整。Git忽略的.schema_extract状态及归档是运行证据，不能为清理正式目录而破坏其路径、调用账或完整性。

## 收口原则

正式正文只描述当前设计、选择关系、来源事实和真实限制；不带本轮过程、模型安排、失败重试或未来待审措辞。完整历史凭据、旧稿、差异、失败检查和讨论放到discussion中；唯一证据先归档保全，确为重复或可再生的内容才删除。

正式目录允许保留简洁、真实的当前来源/manifest/最终检查信息及档案指针。对已变更的正文不能继续让旧receipt冒充当前批准；迁移时保全原字节/身份并更新当前导航、manifest和位置映射。不得去掉源码缺陷、缺失外部比较器、未做GPU验证等实质边界，也不得新增extract/P6批准。

本记录只是留待四算子审修完成后的统一清理，尚未实施迁移。
