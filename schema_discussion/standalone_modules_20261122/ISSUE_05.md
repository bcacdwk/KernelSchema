# Issue 05：掩码语言资料绑定与后续阶段传递

`vector_add-005` 的 Design 已交付正式 Schema；Implement 第一次调用交出完整源码，但明确报告 `needs_input`。原 Workflow 和 Implement 结果均保持 `incomplete / needs_input`，没有候选 GPU 请求或 job，旧结果不改标为成功。

## 真实缺口与官方来源

005 的 root `CONTRACT.Hardware_specification.Feature.unbound_device_semantics` 明确登记 `masked access unbound`，`Wiki.Programming.Language` 是 `<wiki-language@pending:Triton/3.7.1/>`。设计要求 masked lanes 不访存；源码通过 `tl.load` / `tl.store` 的 `mask` 参数处理尾部。Coder 指出未绑定的语言访存语义直接影响越界地址的安全判断。

已从官方 `triton-lang/triton` GitHub API 获取 `v3.7.1`：tag 指向 commit `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`；同提交的 `python/triton/__init__.py` 第 2 行声明 `__version__ = '3.7.1'`。该提交的 `python/triton/language/core.py` 原文规定：

- `load`：`if mask[idx] is false, do not load the data at address pointer[idx]`。
- `load`：false mask 对应值返回 `other[idx]`。
- `store`：`If mask[idx] is false, do not store value[idx] at pointer[idx]`。

这些规则适用于文档规定的 scalar pointer / tensor of pointers；`make_block_ptr` 的 mask 必须为 `None`，使用另一组边界参数。语言层的掩码访存规则不证明周围算术被屏蔽、具体 PTX/SASS、浮点加法舍入或 FTZ、真实设备行为与性能。005 源码对 `other=0.0` 的返回向量执行 `xv + yv`，因此不能仅凭这份访存资料证明设计正文的 padded positions “not added”。

独立材料位于 `schema_workflow/examples/vector_add/triton_3_7_1_language.json`，保留完整原始 docstring、来源 URL、tag/commit 身份、获取时间、文件与 docstring 校验摘要、适用边界，以及上游完整 MIT 版权许可。它不含目标 Schema 或候选答案。原始公开响应、源码、LICENSE、版本文件及核验记录保留于 `.schema_workflow/materials/triton-3.7.1-masked-access/`。

材料 JSON 的 SHA256 为 `c6c0db0d563f16ce70b7a90976706eaafe70ced5c3c63467cffa09d5223c2b7b`；官方 `core.py` 的 SHA256 为 `373eebe68a4c1f6959c81ae6c382b65c5c2893d53274a205f5ff9bbb100fb31b`，Git blob 为 `d424f4984ca34e502591a29a6f0cb525d839c372`。CPU 已核验 tag/commit、响应摘要、三个源文件的 base64 解码与 Git blob/SHA256、逐字 docstring/行号、版本声明和完整许可证；未 import/执行上游模块。

## 正常路径与实际传递边界

DOC2 规定 `<...@pending...>` 不表示正式绑定，依赖入口前应补齐资料，冻结引用使用 commit。Implement 固定 Schema，Optimize 禁止修改 root Contract。因此此次需要新 Design 根据真实资料正式填写语言入口并处理对应缺项，不能把外部 feedback 当作旧 Contract 的覆盖层，也不扩展旧 `needs_input` 的 resume。

只给初始 Design 新增 `--reference` 仍不足以完成后续链路：原 Workflow 不向 Implement 传递 references，Implement 也不自动抓取任意 Wiki URL。后续 coder 因而可能只读到引用身份，读不到必要正文。修复须把普通参考资料固定后传到 Implement，再由它的反馈把实际资料交给 Optimize；Schema 目录或其中的文件仍只作初始 Design 近邻。

传递还须区分补充资料与候选反馈：历史 Optimize/Implement 反馈含原候选的 Schema/kernel 摘要和相对路径，不能无条件并入当前 Optimize 的候选证据。材料存在、已快照或出现在提示中，也不等于模型已经阅读或候选已经通过验收。后续正式模型与固定全案例 GPU 验证由主代理按正常入口继续。

## 最小修复

Implement 新增可重复的首建参数 `--reference`，与 `--feedback` 分开固定。这里选择明确接口，而没有根据文件名猜测材料类型，也没有把所有历史修复反馈自动加入当前候选材料。resume 仍只接受追加 feedback，不能更换 references；原 `needs_input` 终态规则不变。

Workflow 使用既有 root/dispatch 文件边界识别 Schema 参考，不解析 Schema 内容。普通文件或资料目录在每个新版本的第一次 Implement 时通过 `--reference` 透传；同版 repair resume 复用原快照。Schema 目录及其中的文件仍只给初始 Design 作近邻。原 Schema 起点仍拒绝 `--reference`，原 config/input 摘要检查和累计次数规则保留。

Implement 将 reference 原字节放入 `inputs/reference/NNN/`，保留目录拓扑、原位置、逐文件与整体摘要；初始化即检查下游闭包识别的 JSON 路径字段、Markdown 链接与可解析本地引用。普通单文件须自足；有本地依赖时提供包含它们的完整目录。缺失、绝对或逃出该资料范围的链接，以及需要完整父目录但只给了一个文件的情况，均在 coder/SDK 之前返回 `needs_input`。带顶层 Schema/kernel 摘要绑定的 JSON 明确属于候选证据，要求使用 feedback 通道，防止历史摘要混入当前候选验收。没有自动网络抓取或通用资料重定位。

每个新 `feedback.json` 的 `materials` 只列这组已固定 reference 快照的相对路径和上述来源摘要，不导出 repair feedback。真实 Optimize 的现有 closure 因此可以读取同一份固定正文与目录内相对依赖，无需修改 D/O。Implement 提示明确区分源资料、历史反馈与本次观测，并强调 Task/Schema 权威以及 pending 绑定不能被资料隐式覆盖。结果字段、原请求/job 恢复和次数规则不变。

## CPU 验证与交接

使用既有独立 venv 的 Python；测试进程的 CLI 配置目录是临时空目录，不读取正式认证。未启动模型或 GPU。

```sh
python -m pytest -q schema_implement/tools/tests/test_implement.py schema_workflow/tools/tests/test_workflow.py
# 63 passed in 23.45s

# 补充单文件不能代表整个父目录的边界后，只重跑受影响路径与完整装配：
python -m pytest -q schema_implement/tools/tests/test_implement.py schema_workflow/tools/tests/test_workflow.py \
  -k 'reference_boundary or reference_directory or real_workflow_implement'
# 8 passed, 56 deselected in 6.21s
```

正常 Workflow 链测试使用真实 Implement/Optimize CLI、参数解析、快照、request 构造、feedback、Optimize closure 与提示装配；Design 交付、模型回答和 GPU 结果是明确的 CPU fixture，不声称真实设计/设备验收。fixture 实际读取固定资料，确认官方 JSON 原字节传至 Optimize，完整资料目录内相对链接仍可读，原位置/摘要未丢，Schema 近邻未误传，Task/eval 与源资料未改。另覆盖新版本重新传材、同版修复不重复加入、resume 禁止替换、固定快照篡改拒绝、repair feedback 不混入 materials，以及非法资料在 0 次调用时被拒。

对计划用于 006 的 env、官方语言 JSON、自足续写说明和 005 的单文件 kernel 源码，已执行仅 CPU 的 `snapshot_reference` 前置核验，四份资料全部通过；记录在 `.schema_workflow/materials/triton-3.7.1-masked-access/continuation_preflight.json`。语言 JSON 原字节及其摘要保持不变。续写说明保留真实缺口与旧反馈摘要而不导入旧候选反馈 JSON。新 Design 仍须自行作正式绑定，后续真实全案例结果由新运行决定。

本 Issue 与模块修改交主代理整合提交；根方法/README、示例说明及共享 EXECUTION 由主代理维护。005、固定 Task/eval、受保护目录和 D/O 代码未修改。
