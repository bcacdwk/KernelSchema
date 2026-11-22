# 第二轮 D：独立复核、兼容与公开收口

D 的必需项已完成。W2 有限独立复核和 current 终检通过；workflow 187 个不同定向用例均已有对最终适用字节有效的通过证据。精确 stage 清单由 HISTORY_PUBLICATION 与本地 round2_d/stage_allowlist.json 给出，Git 提交和正常推送交根代理执行。

[W2_REVIEW](W2_REVIEW.json) 固定 SHA `af96e504979f3a98cd975f9f82922ad088821ad4a0c13ed0e217ea8d45679255`，只绑定 subject `4936e08c5611c703497c6e53a244ff4e3cf746f993ebff62dce57c60752a2242`、来源与 checker，不包含后来生成的 current 摘要。复核覆盖九 YAML、两 dispatch、19 YAML/9 dispatch 完整目录、五个固定 ACCEPTANCE 的九输出/十有效节点和真实隐式 root/显式 from 闭包。两处发现由 C 同轮更正：GDN 显写缺 CTA 聚合风险；mixed Sparse 区分缺原题回执和题目哈希未确认。受影响检查、subject 和独立证据刷新后才允许生成 current。

[九 current 与索引终检](round2_d/current_verification.json) 逐项确认 files/Version/Source/源 SHA、依赖、实际规则/有限审阅及原 receipt。九个实际目录共 27 个 bridge 断言通过：旧 receipt 拒新字节，curated+receipt=None 接受，current 作为 receipt 拒绝。原 receipt/DELIVERY_INDEX 和其他 24 YAML 未改；新 current 不承担 P6 批准。

[workflow 验证](round2_d/workflow_verification.json) 使用指定 pytest 环境、显式 main PYTHONPATH，Template 私有命名空间来源实测在 main，PATH 无真实 claude/codex。初次六模块 187 例为 154 pass/33 fail：32 例受示例 DOC1/4/5 旧材料 pin 阻塞，一例是 D 新增 Rules 反例未排序 refs。根在 [OWNERSHIP_CHANGES](OWNERSHIP_CHANGES.json) 精确分配 materials.json 后，只更新三条 size/SHA，并修该合成 fixture。70 个受影响用例先有 69 通过，余一原 strict CLI 预算夹具补上显式 --writing-mode strict 后单例通过（默认 write_only 没有该 reference profile，不改产品默认行为）。117 个不受影响通过结果按文件摘要/函数 AST 和依赖核对复用；不把重跑计为额外独立覆盖。初始完整失败日志留本地，公开修正摘要保留失败原因、原日志 SHA 和真正的后续通过。

没有改 kernelschema 的 parser、writer、bridge、api 或 iterations。默认 write_only 的单次 writer/原文保留，无隐式 reviewer/效果/GPU gate；strict 新冻结规则的真实 BUDGETS=300/2、旧规则漂移拒绝；标准 AKA lazy mock 委托与 iterate 的 keep/失败反馈/未结算停止/不重放均保持。文档材料 pin 是新版本公共输入更新，不迁移旧 Rules 或账户。

[导航核验](round2_d/navigation_verification.json) 为 18 项/48 YAML，原六行导航逐字保留；本批十二项区分九 current 和三原 receipt。[保护核验](round2_d/preservation.json) 包括 695 原件、277 允许保护文件、12 state/12 receipt/supervisor/AGENTS；42 archive 元信息先排除，不访问目标。旧 57 文件已从原基线提交逐 SHA 还原核验，workflow 37 个自然提交仍可达。

[实施覆盖](IMPLEMENTATION_COVERAGE.json) 收齐 26 发现、24 候选/104 出现、20 recap、11 既有修正、16 删减和 FC01/02 的实际路径、测试或保全证据。五个旧候选全文审阅通过；其余 690 原件保持 local-only，少量有限来源摘读与未内容审阅的捕获分别标明。主入口仅导航公开材料；fresh clone 的本地历史、原 runtime、/tmp 环境和禁 archive 来源边界见 [公开历史](PUBLIC_HISTORY.md)。

初始文件定位命令有一次范围偏宽的目录级探测可能，已在 [过程边界](round2_d/process_boundaries.json) 如实记录；未打开 archive 正文，archive 不构成验证来源。

所有 fixture 使用新临时目录；未运行模型/GPU/真实 Extract/monitor，未改 AKA、固定来源、旧 run/账/receipt 或 AGENTS，未做 Git 写操作。masked 4616 索引组合是旧证据复用，不是新测试；未证明全 33 稿、全域数值、编译器寄存器分配、来源风险消除或真实模型成本收益。没有本轮必需项遗留。
