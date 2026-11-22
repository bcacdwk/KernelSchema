# Issue 01：Design 未交付可核验来源与完整可读路径

`vector_add-001` 的首个 Design 写作调用（`drafts/02.json`）在 900 秒后退出 143，无最终回答或结构化稿。公开读取记录显示 DOC1–6、Template README/starter、Task 绑定 eval 文件均已提供；模型另反复搜索 `Golden_reference`、commit 与硬件来源，尝试读取未授权 checkout/Template 根目录而失败。

自然 repair 第三调用也在 900 秒后退出 143，无结构化稿；公开最终文字仅为 `I have everything I need. Writing the complete collection now.`，并再次尝试被拒的 Template examples 与目标示例 Glob。Workflow 最终 `incomplete/write_failed`、Design `calls_used=3/schema=null`；i04 确认全部相关进程退出、没有 GPU job 后才切换产品写权。原失败目录保持不变。

已确认的材料缺口：manifest 原来只有原路径、快照路径、SHA-256 与大小，没有 Task/reference 的真实 Git 身份；规范要求的 `repository@commit:path::entry` 无法单凭该清单核验。复制的 DOC/Template 目录名称也与原文相对链接不一致。原有 Task/eval 六文件快照逐字节匹配 `4f7919bf531ad8b7ab49d5e371dfc8b7fea51a34`，其中 reference 与材料提交 `2c9c346cdd0373821b633012bdfece446c766d65` 的 blob 一致。

修复限定在 Design/Optimize 各自的 runner、prompt 和 CPU 测试：

- host 检查本地 commit tree 与 blob，只有 committed blob 与固定字节完全一致才附加可用 locator；已修改、未提交或 Git 不可用时明确状态，不自动提交输入，不查询或记录 remote URL，不复制 Git/认证目录。
- 将事实送入 manifest 与对应材料的请求头；明确原路径仅作来源上下文，模型使用已固定快照。
- DOC/Template 保留原有相对目录结构，只额外固定一个现成完整 QSA 格式案例的 README/YAML/dispatch（44,026 bytes），仅提供可读路径，不塞入正文或目标 vector_add Schema。
- 模型仍仅能 Read/Glob/Grep `input/`，900 秒和调用计数规则不变。缺少必要硬件/编程事实时具体报缺项，不把历史示例环境当本次观测。

公开材料能证明上述交付缺口，不能证明它们解释了全部超时，也不能由 CPU 检查证明 Qwen 现在必然完成；真实行为须由下一次经交接的新运行验证。现有格式案例没有已绑定的 Triton 硬件/编程手册，本次没有编造或补写设备事实。

应用后的受影响 CPU 检查：

```sh
/home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python -m pytest -q schema_design/tools/tests schema_optimize/tools/tests
# 32 passed, 8 subtests passed in 23.74s
```

新增四项测试各覆盖 Design/Optimize，验证真实 commit 字节与带特殊字符路径、同长度修改及未提交文件、Git 不可用、凭据不进入材料、文档相对链接、格式案例可读范围。另用真实 Task 只运行材料准备：20 个固定文件中 Task/eval 六项均 `git.status=verified`，reference blob 为 `d8c4d059869ef22c7f71e1e6f9756f94a0a02bb8`，真实 locator 已进入请求，`calls_used=0`。没有读取目标 vector_add Schema。

`git diff --check` 通过；保护区、固定 Task/eval、Implement 和 Workflow 的 diff 为零。未 stage/commit/push，未启动新真实模型/GPU。后续由实跑负责人用已交接的新产品验证，不能复用已耗尽的 001 写作次数或改变其失败结论。
