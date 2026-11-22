# B15 / standalone Demo reuses the original parent

2026-11-21。基线 KS `204cde57b3d80426bc5367ffa186f69669513f59`，唯一源码工作树 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1`。本代理未提交、推送或派生代理，未访问/枚举/修改/运行 `AKA_experiment`，没有安装依赖或下载解释器。

本批为实际 standalone driver 增加显式复用已有 parent 的入口。诊断可先使用原 OwnerAccount；之后冻结 Task/remote 时，新 Demo 直接连接该原账户，不再强制创建新余额。默认不传外部 parent 的行为仍自建一个 owner。此处仅交付工程接线与 CPU 证据，新增真实模型调用、GPU jobs、网络请求均为 **0**。

## 接口与边界

仅四个分配文件有源码改动：

| 文件 | 交付 |
| --- | --- |
| `examples/workflows/row_softmax/driver.py` | `Demo.create(..., parent=original_ref, parent_bundle=Bundle(original_bundle), authority=original_authority_path)`；三项必须一起显式提供。`parent_bundle` 也接受明确目录。 |
| `examples/workflows/row_softmax/run_demo.py` | `prepare` 新增 `--parent-ref`、`--parent-bundle`、`--authority`。Ref JSON 仅含原 Ref，不新增通用 handoff 协议。其他命令不接受替换 parent/authority。 |
| `tests/workflows/test_standalone.py` | 15 项已有账户定点，用独立 synthetic 临时账户和运输；保留原默认入口测试。 |
| `docs/workflows/standalone.md` | 三输入 Python/CLI 用法、原预算与恢复/导出边界。未声称真实 worker 或模型资格通过。 |

创建新 Demo 前，先用 Bundle 读并验证原 Ref 的 canonical `budget_account`，用 `OwnerAccount` 公开构造器只读打开显式私有 authority，再比较完整 descriptor（账户 ID、owner、原限额与截止均包括在内）。调用方省略 `limits` 时保留原限额；显式传入时必须与原限额 JCS 完全一致，冲突在输出目录创建前拒绝。

只用 `import_bundle` 导入所选原 parent Ref 闭包，保留原路径和字节。该分支不调用 `OwnerAccount.create`，不复制账户文件，不迁移余额。原 operations/历史消费、暂停、fence 和唯一活动时钟仍由原 OwnerAccount 管理。新 Demo 不注册诊断替身，也不建立第二个账本。

`.runtime/demo-anchor.json` 以 `0600` 保存必要的原 authority 绝对路径；公开 config、请求、summary 和 selected export 没有该定位或私有账户内容。恢复严格读取 anchor 定位，不从默认位置或环境寻找替代账户。定位丢失/非绝对、authority 缺失或 descriptor 不匹配时拒绝，不创建新余额。原 driver implementation hash、scope/source 和 child run 身份规则不变；旧真实 run 不自动迁移到本版。

后续仍使用实际原 driver 的 `prepare_agate`、`run`、`continue` 及 `kernelschema.api`。调用者可以在 remote 未知时传 `remote=None` 准备 Task/Q；这不会建立 GPU 资格。明确真实参数就绪后应在新的、尚未准入的 Demo 上冻结参数，不能改写已准入原 profile。

## 定点证据

解释器为已有 `/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python`，Python 3.13.12；未运行全库。

| 轮次 | 结果 | 日志 |
| --- | --- | --- |
| 新已有账户组 | **15 passed, 16 deselected / 82.72s** | `/tmp/schema-b15-existing-parent.log` |
| 原默认两轮与 fixed/partial/verify 五请求入口 | **4 passed, 27 deselected / 70.71s** | `/tmp/schema-b15-default-compat.log` |
| 三个修改 Python 文件 AST、四文件 `git diff --check` | 通过 | 本轮终端记录 |

命令均在指定 KS 工作树执行：

```sh
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B -m pytest tests/workflows/test_standalone.py -q -x -k existing_parent
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k2-cpu/bin/python -B -m pytest tests/workflows/test_standalone.py -q -x -k 'two_rounds_real_api or five_requests_and_explicit'
```

新增组覆盖：

- 创建前已有 synthetic coder1/evaluator1 的原历史，创建后私有 authority 字节完全不变；测试将 `OwnerAccount.create` 替换为失败函数，确保所有后续分支零建账。
- 实际 driver 的六步 synthetic Agate qualification，加 design→implement→optimize→implement；写作发布点模拟崩溃后重新构造 Demo 并继续。每个原 child 的公开 parent Ref 和私有 budget binding 都指向同一原账户；原历史 operations 未变，账户最终累计 synthetic calls9/jobs9。
- 默认 Task starter 的两次 implement 都保持 `seed=None`，新 Schema 与旧 Schema 不同，原 API 选定动作真实执行；完成后再 continue 没有重复运输。
- public parent 原 Bundle 重命名后仍能继续，因为准确公开 Ref 已导入；私有 authority 原位置必须保留。导出保持原 public Ref 字节且不含 `.runtime`、authority 路径或原账户文件内容。
- 创建前 pause、unknown fence、仅剩不足 design 最低向量的 model_calls 预算分别保留；不新增模型运输。fence 阻止注册新请求；原未知 operation 活动时间继续累计。
- 缺三输入之一、预算冲突、不同 owner/账户、错误 Ref hash 在写输出/改原 owner 前拒绝；恢复缺定位、相对定位、缺 authority 和错误 owner 均拒绝且不补建本地 owner。
- CLI prepare/inspect/export 的原账户定位、公开结果不泄露私有定位和三输入不齐的失败路径。

首轮快速测试记录在 `/tmp/schema-b15-existing-parent-fast.log`：1 passed、1 failed（9.86s）。失败原因是测试错误要求 fenced 原请求的完整状态字典不变，其中 active_ms/active_ns 本应继续累计。仅修正测试为请求集合/身份/限额不变、active 单调不减；生产预算与计时语义没有放宽。随后上述最终 15 项全部通过。

## 冻结交接

四个文件的最终 SHA-256、mode、size 在 `/tmp/schema-b15-demo-existing-parent/source-snapshot.json`，清单 SHA-256 为 `91fa3a6408345604be415a8e43a17be6ff5f8c3b2bda1e2089863ee7c0b3c35b`。仅归属四文件的 patch 为 `/tmp/schema-b15-demo-existing-parent/owned.patch`，SHA-256 为 `d80d79dc47ab4e9812ef9b603855e307cc74ee1d8a69898550fe70de370944c9`。

| 文件 | SHA-256 |
| --- | --- |
| `examples/workflows/row_softmax/driver.py` | `a46fc7924d48be2a274be496dd75a8d32e4cceda443ff90736a5081e10b9e80b` |
| `examples/workflows/row_softmax/run_demo.py` | `a3c9e692a3d53c319b817150aaa545d823a1b682d5e95c3b83b695d99d97e9ed` |
| `tests/workflows/test_standalone.py` | `2d09506f084faf83a02067b2b92ff23bc75c3115647d548b6ad30405a02f640d` |
| `docs/workflows/standalone.md` | `9091ef87dc70e247389076e0b60b426fd8304f9e5c88d449b3b6c0aeb907e3c4` |

本代理只读过指定 B14 handoff 来理解公开字段，没有打开、启停、修改或消费其真实 authority/operations。B14 原诊断 HTTP400 `dev_cluster_disabled`、入队前拒绝、jobs0/model0/active531ms、fences=[] 是 B14 owner 与主代理提供的实际观察，本批不重新核账或调用。认证 HTTP200/缓存目标信息不等于 worker dev 权限或完整执行资格；GPU 后续阻塞由 B14 报告记录。本补丁不修改 evaluation、目标材料、原 R0、既有真实失败账户或其他仓库。

上述四文件已冻结供主代理精确提交；本报告位于原 main 的唯一授权报告路径。新 Git 提交 SHA 由主代理签收提交后记录。
