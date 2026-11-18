# origin 合并记录

2026-11-18 UTC。按用户要求先合并远端，再交回既定 Extract 实施；本次只完成 Git 合并与短检查，未启动提取、模型调用或 monitor，未实施无 timeout 方案。

## 结果与远端范围

- 分支：`main`；原 HEAD：`fa853fafd32a78ac496fbf71e615e34200eeeae3`。
- 新 HEAD / `origin/main`：`f48823bdea07aaadd2ccff7678589d20701cbf82`，提交 `markdown and readme refine`。
- fetch 后本地领先 0、落后 1，采用 `git merge --ff-only origin/main`；没有分叉、额外合并提交、历史重写或推送。
- 远端修改 41 个文件：34 MD、3 Python、2 YAML、2 PNG，424 行新增、491 行删除。`AGENTS.md` 未变。
- 3 处 Python 改动均清除已删除的 `EXECUTOR_ALIGNMENT.md` 引用，与已有本地修复一致。两个案例 YAML 及 PNG 原样保留远端版本：FA4 `packed_persistent` 明确 `2*cpad`；QSA 明确 ticket/base、修正投票及 halfmax 覆写边界。

## 真实冲突与解决

合并远端本身是快进；随后对保存的本地 WIP 执行精确 `stash apply`，产生以下 7 个内容冲突。这里 stage 2 / `Updated upstream` 是新远端，stage 3 是旧本地 WIP；逐文件以新 HEAD 为底处理。

| 文件 | 解决方式 |
| --- | --- |
| `EXTRACT_SCHEMA.md` | 完全采用远端；远端已覆盖本地失效链接修复，并保留新阶段、维护与发布说明。 |
| `README.md` | 完全采用远端；不插回本地旧段落或重复的链接修复。 |
| `schema_extract/README.md` | 两处冲突均采用远端；仅补回现行 acceptance 类型与输出参数事实，原有远端文字不改。 |
| `schema_extract/prompt/phases/01_contract.md` | 保留远端完整公式/局部标签边界原文；另加现行 `comparisons` 字段的精确键名与证据绑定说明。 |
| `schema_extract/tools/README.md` | 保留远端 raw、workspace 与裁决候选路径说明；仅删除与 LIVE15 代码相反的“每段最多 12 行”。 |
| `schema_template/README.md` | 完全采用远端；已有本地链接修复由远端覆盖。 |
| `schema_template/examples/README.md` | 完全采用远端；保留远端精修的验证边界与来源限制。 |

其余重叠文件自动合并：`schema_extract/REVIEW.md` 与远端一致；`prompts.py`、`snapshots.py` 的本地修复已包含在远端；`test_prompt_materials.py` 保留其他本地回归用例。没有代码、YAML 或 PNG 冲突，也没有必须由用户决定的未解冲突。

## 相对远端保留的文字事实修正

下列是全部活跃 MD 差异；不包含新建的本合并记录。

1. `schema_extract/tools/README.md`：删除“每段最多 12 行”，其余远端原句不改。实际回执验证已在 LIVE15 删除任意跨度上限，仍核验真实文件边界、offset 与摘录。
2. `schema_extract/prompt/operations/response.md`：同样删除旧 12 行上限，保留本地对两类证据选择必要原文范围的说明；与真实回执契约一致。
3. `schema_extract/prompt/operations/check.md`：补回 P1–P5 worker 以规范化预览决定退出码、reviewer/P6 仍检查 raw 的事实，避免把等待 Runner 落地的机械 warning 当作失败。
4. `schema_extract/prompt/phases/01_contract.md`：仅新增 `comparisons.output/source_output` 精确名称与 JSON 声明证据索引要求；不改远端“不能从局部标签推组合公式”的段落。
5. `schema_extract/README.md`：新增 acceptance 中 JSON、比较器代码、Markdown 的解析边界；代码/说明中的同名文本不会令整文件按 JSON 解析。
6. `schema_extract/README.md`：新增当前仍存在的 `--max-output-tokens` 表项及传递/冻结说明。它描述合并时实际代码；后续移除该参数时再据实现最小更新。

对远端既有原句的修改仅是前两项旧 12 行限制；其余为必要事实补充。没有其他文案修正，没有新增规则或推断远端未写出的要求。

## WIP 与恢复备份

原工作区为 33 个 tracked 改动、39 个未跟踪文件，index 和 stash 列表均为空。先备份原 HEAD/refs、index、状态、完整 binary diff、dirty/untracked 归档与 SHA-256 清单；随后用唯一命名 `stash -u` 保存，快进后执行 `stash apply`，未使用 `pop`。

- 保留的 `stash@{0}`：`aba821f7d9717c6761ca66132fb07fedc039534e`。
- stash 名称：`origin-merge-20261118-fa853fa-preserve-extract-wip`。
- 独立备份目录：`/home/shaohanyong.shy/.codex/backups/KernelSchema-origin-merge-20261118T043633Z-5xazyis4/`。
- 该目录保留 `repository.bundle`、`wip-stash.bundle`（均通过 bundle verify）、原 index、`worktree.patch`、`tracked-dirty.tar.gz`、`untracked.tar.gz`、原始文件清单、冲突原稿及解决后验证记录；另保存合并后的 patch 和本报告副本。当前工作区已恢复，勿直接重复 apply。
- `.schema_extract` 未纳入 stash/归档，也未删除或改写；仅比较其 64,214 项路径、mode、size 与纳秒 mtime，前后完全一致。

最终保留 26 个 tracked 未暂存改动，原 index 空状态已恢复。数量减少的 7 项是远端已覆盖的链接/清单修复；全部 23 个原 tracked Python 文件与合并前 WIP 的 SHA-256 一致，原 39 个未跟踪文件亦全部一致。新增本报告单独保留，既有讨论和试跑历史未改写。

## 检查与后续实施

`git diff --check` 通过；index 无冲突或暂存差异，变更文件无冲突标记。37 个远端改动文件与新 HEAD 完全一致；其余 4 个仅含上述三个文档补充/修正和原 `test_prompt_materials.py` WIP。97 个规则文件全部存在，活跃入口不再引用已删除执行器文档。未运行全测试、提取或在线验收。

既定实施范围无须扩大，但须以新 HEAD 的精修文字为底，仅按最终行为更新时间、输出与巡检说明；保留新的维护 `fields` 上限、ready/发布区分、standalone 范围、review-eval 标量复制任务边界及 ack 已确认后仍 scan 的语义。P1 的组合公式不可从局部标签自行推出。案例 YAML/PNG 和历史报告保持本次合并后的版本，旧 run 冻结快照不升级。
