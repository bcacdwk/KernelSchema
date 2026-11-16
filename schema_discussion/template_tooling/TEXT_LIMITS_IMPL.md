# 字符／形式实现交接

范围以 `TEXT_RULES_PROPOSAL.md` 最新表为准；未修改 examples、wiki、Extract 或运行状态，也未改 demo。

- `text_policy.py` 共享 Unicode／逻辑行计数及最小数值形式、可解析单名检查；原预算保留，补 Stack 150/1、retention 150/1、Work 指定字符串 300/2。
- `values.py` 按值分支选预算：名字 50/1，普通 dtype／Storage.location／硬件登记键 100/1，整数表达式与固定值字符串叶 100/1，内部 binding 和 source 摘要 300/2；Input 150、unknown 150 保留，不扩大类型。
- `pipeline_syntax.py` 只补单名／单表达式检查，事件和 Waiting 无整串预算；Unicode 数学、值的 ∞／∅、事件 `*` 和省略 slice 端点保留。
- `structure.py` 核对 retention 的既有 invocation／括号实例与可选 `; i=lower..upper` 外壳；`symbols.py` 先匹配已有完整名称／路径，再检查弱引用外壳，300/2 不统一禁换行。
- `local_repository_errors(files, documents, skip_hardware=...)` 仅对明确同本地 Git 仓库、且 ref 可用的真实定位符核对该 commit 的路径；尖括号 Wiki、未知仓库／修订不报存在性错误。文件入口 hook 由布局代理接入，纯快照 API 保持零 I/O。

最终验证：整合并经独立审阅修正后，完整合成套件 82 项通过；具体结果见 [TEXT_RULES_REVIEW](TEXT_RULES_REVIEW.md)。

真实边界：不从 math／strategy 建伪符号表，不证明数学整数性；保留无法可靠区别于真实名称／路径的说明，未解析语义目标和源码入口含义仍由审阅核对，不添加 NLP、联网或通用源码解析器。
