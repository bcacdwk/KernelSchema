# B38：简化入口文档与固定源码配对

2026-11-21 UTC。按 B37 的最小交付缺口修订默认 write-only、AKA standard 安装及示例说明；不增加评测器、初始化平台或新质量门。本 agent 未提交、推送、安装/构建包、调用模型/GPU、启动服务、派生子代理或访问禁止实验仓；没有干预 B36 的固定执行环境。

**文档、真实 metadata 示例和 KS 配对内容已完成；AKA 精确提交与推送交给根。** 原 AKA 执行基线是 `621481a9b8685f4880e32a9ddce56f4e3d94cd07`。根确认最终 KS 提交 `907c99aa1589bddf73e26079f2304b83d68a8e51` 后，本 agent 将具名子模块 detached 到该 SHA、用原 freeze 工具更新 release。前一文档提交 `6e9264e` 从真实 public origin fetch 成功；最终提交的远端 fetch 持续无输出，只精确终止本 agent 的该 fetch 后，按授权从 KS worktree fetch 同一固定 Git 对象成功。保持真实 origin，未安装 B36 环境或创建新源码仓库。

## 入口和安装

- KS [schema_implement/README.md](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/schema_implement/README.md) 首段现为 `schema/problem/repo_root/platform/workspace` API 与对应 CLI，直接调用原 AKA standard。旧 request/continue/reference 明确放在 legacy 小节，失效 `#agate` 链接替换为实际兼容文档。
- KS [protocol.md](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/docs/workflows/protocol.md) 说明任意 UTF-8 task 文件的 write-only；row-softmax 限制只属于 strict Task adapter 与固定示例 driver。标准 native 结果与旧五项 Source qualification 分开说明。
- KS [standalone.md](/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1/docs/workflows/standalone.md) 添加 AKA 安装与真实 Task baseline 的短链接。两处跨仓文档链接已修正为实际 fork 的 `feat/schema-workflows-v1` 分支；[public-doc-link.json](/tmp/schema-b38-simplified-docs-pair-20261121/public-doc-link.json) 记录 HTTP HEAD 200。
- AKA [application-plugin-migration.md](/home/shaohanyong.shy/atrex-kernel-agent/docs/application-plugin-migration.md) 显式构建原 Framework/Legacy/Light/Dashboard/Observation 五包，并使用新临时 wheel 目录，避免通用 `*.whl` 安装意外包含 Schema。
- AKA [schema-standard-workflow.md](/home/shaohanyong.shy/atrex-kernel-agent/docs/schema-standard-workflow.md) 复用现有 builder，先安装匹配 KS source/editable，再安装同 checkout 的 Framework/Legacy/Light/Schema 四包。说明 `--no-deps` 避免替换所选 KS，`--force-reinstall` 处理仍为 `0.2.0` 的新源码；已有 torch/CLI/后端 runtime 前提独立列明。版本号不是 source identity。
- AKA [schema-optional-module.md](/home/shaohanyong.shy/atrex-kernel-agent/docs/schema-optional-module.md) 将安装统一指向标准说明，仅显式 legacy `--schema-config` 继续调用既有 manifest verifier。没有把 verify_release/transport qualification 设为 standard 门槛。

## 可直接使用的标准示例

原 [vector-add 示例 README](/home/shaohanyong.shy/atrex-kernel-agent/skills/atrex-aka-light/examples/native-atrex-bench-vector-add/README.md) 说明：三文件足以 Task load，但普通 full evaluation（包括 leaderboard）需要逐 shape 实测 metadata；real Bench source 不能用空 marker 代替；Shape keys 必须可转整数。

标准文档给出原 `agate eval --candidate reference.py --reference-dir ... --gpu L20D --shape-shards 1 --bench-iters 30 --job-timeout 600 --no-wait` 和同 job 的 `agate get ... --wait` 命令，以及 `metadata.shapes[ID].production_performance.performance_us`、`perf_gpu` 与 job/input provenance 的现有格式。没有新增 baseline 转换器或验证平台。

附入示例的 [metadata.json](/home/shaohanyong.shy/atrex-kernel-agent/skills/atrex-aka-light/examples/native-atrex-bench-vector-add/metadata.json) 是 B32 原件逐字复制，来源为 `/tmp/schema-b32-standard-real-20261121/numeric-shapes/atrex-bench/data/native-atrex-bench-vector-add/metadata.json`，SHA-256 `17d5b6e61678e635bb168f3afa7091237709f5fbf22f7d30c7c13f4eefa9f740`。只复制公开 Task metadata；没有复制私有账本、认证或 capture。

| 输入/结果 | 已核事实 |
| --- | --- |
| 原 job | `ev_1861beac3248`，2026-11-21 UTC，NVIDIA L20D，all_pass |
| Task | 连续 bf16，`x + alpha * residual`；IDs `0/1/2`，N/alpha 为 `1024/1`、`65536/0.5`、`1048576/2` |
| 原实测 latency_us | `9.456000290811062`、`11.215999722480774`、`11.264000087976456` |
| 测量主体 | 原 PyTorch reference 自测；不是优化 Triton、SOTA 或生产 kernel |
| 远端 runtime | `source=job_install`，`atrex-bench==0.1.0.post20261111022558`；evaluator SHA `183b6288b85f0cd7ee3828c1a2a9db922f8971774e0122f5fceccfb58e09456e` |
| 本地 layout 来源 | `alibaba/atrex-bench@1fa1f0d8a663093a22946270e7d6accb00c33698`；没有冒称它等于远端 wheel Git revision |

示例 README 列出上述来源与四文件 SHA；reference/input/shapes 也与 B32 原始 Task 逐字一致。`production_performance` 是原字段名，不提供生产优化结论；换设备、输入或重新比较时使用实际新测量。

## 有限验证与实际 kernel 状态

[check_docs.py](/tmp/schema-b38-simplified-docs-pair-20261121/check_docs.py) 与 [docs-verification.json](/tmp/schema-b38-simplified-docs-pair-20261121/docs-verification.json) 记录：45 个本地文档链接及锚点有效；文档 shell/Python 片段只做语法检查，未执行安装或运行命令；四份 CLI help 均包含所述参数。help 通过独立进程的临时 `PYTHONPATH` 读取源码，使用原解释器和 B32 固定 SDK；没有安装到活跃环境，不能称为一次新打包安装验证。

原 Task `validate_performance_baseline` 的纯定义在 CPU 上读取新示例 metadata，返回与 B32 summary 完全一致的三个浮点值；没有修改数值语义。KS、AKA 的 `git diff --check` 已通过。B34 60 个、B35 16 个针对性 CPU 用例按原报告范围复用，没有重跑或冒称为本 agent 新跑。最终链接补提交只改两篇文档；没有重跑无关 help/运行检查，重新核对了固定来源和原 manifest verifier。

标准结果说明新增：`completed` 或版本 `PASS` 应结合 `candidate_commit`、`measurement_subject`、`measurement_source` 与原候选 evaluation/promotion 阅读。B35 的 `canonical_incumbent_carry_forward`、subject incumbent、candidate_commit null 仍只是 V0 继承值，不能称新 kernel 成功。**B38 没有建立新的 kernel 成功证据。** 根本轮消息另确认 B36 已终态：V0 all_pass，但 coder 900 秒超时、没有新候选，workspace resources CLI/gpu-wiki 出现 `plugin_changed`。此处仅转述根的终态说明，未读取 B36 环境或自行复核；完整证据由根/B36 报告提供，不记为新 kernel 成功。

## 最终待提交源码配对

- KS 固定提交：[`907c99aa1589bddf73e26079f2304b83d68a8e51`](https://github.com/bcacdwk/KernelSchema/tree/907c99aa1589bddf73e26079f2304b83d68a8e51)。具名子模块保持 `origin=https://github.com/bcacdwk/KernelSchema.git`、该 detached HEAD 与 clean 工作树。
- 本 B38 在 AKA 基线 `621481a9b8685f4880e32a9ddce56f4e3d94cd07` 上仅修改六个普通文件和 `optional/KernelSchema` gitlink，待根精确提交；其他代理并发修改的运行代码不包含在本清单或补丁中。子模块工作树已更新；父仓库 index 在交接时仍为旧 gitlink，根须精确 `git add optional/KernelSchema` 与下表文件，不能只提交 release。
- 原 `tools/freeze_schema_release.py --commit 907c99aa1589bddf73e26079f2304b83d68a8e51` 生成 **300 个选定文件**。`release.json` SHA-256 为 `0bdd28416167fbcb73ce62c6ffcb2a0a9d338eaf5a02a8ab21469d666de7375f`；api/cli/writing/claude 等新接口与 retention 源码均匹配固定 KS。现有 `aka_base_commit` 仍明确是历史 integration base，不冒称当前 AKA 包身份。
- 既有 `verify_release()` 在独立 source-PYTHONPATH 进程通过，[source-release-verification.json](/tmp/schema-b38-simplified-docs-pair-20261121/source-release-verification.json) 指向实际子模块 root 和同一 manifest。此为本次 provenance 核对，不是 standard 的新增准入门。首次在 checkout 尚未结束时调用 freeze 被原 HEAD 前置检查拒绝、未写 release；待 checkout 完成后上述原命令成功，未放宽检查。
- 精确清单为 [source-manifest.json](/tmp/schema-b38-simplified-docs-pair-20261121/source-manifest.json)，SHA-256 `74b9a6e563a49d8c954959f52065a668ca4f216311c42417109f0585bfffcb33`；普通文件副本在 [frozen-aka](/tmp/schema-b38-simplified-docs-pair-20261121/frozen-aka)，补丁在 [aka-source.patch](/tmp/schema-b38-simplified-docs-pair-20261121/aka-source.patch)。新 metadata 为 0644，内容字节保持 B32 原件。

| AKA 文件 | SHA-256 |
| --- | --- |
| `docs/application-plugin-migration.md` | `2bf2468faf785f9228962955e34fcf7d4add747a6b89bc61de5d80121b60cc00` |
| `docs/schema-optional-module.md` | `67531d07cdbcbf7a9055bcb5fa2c2444d5b9e5547fa4e067e727213f9367fb3f` |
| `docs/schema-standard-workflow.md` | `34d94ff508b935bfc3bc3a3e4efffbc3ff92c5aa6db009f56cfa040892597e66` |
| `skills/atrex-aka-light/examples/native-atrex-bench-vector-add/README.md` | `6021b09e6b5fddbd582f5d9f586d4096d336b63a4592f13575b7b811b63a9c8e` |
| `skills/atrex-aka-light/examples/native-atrex-bench-vector-add/metadata.json` | `17d5b6e61678e635bb168f3afa7091237709f5fbf22f7d30c7c13f4eefa9f740` |
| `src/aka/workflow/schema/release.json` | `0bdd28416167fbcb73ce62c6ffcb2a0a9d338eaf5a02a8ab21469d666de7375f` |

KS 三文档逐提交核对见 [ks-docs-fixed.json](/tmp/schema-b38-simplified-docs-pair-20261121/ks-docs-fixed.json)：

| 文件 | SHA-256 |
| --- | --- |
| `schema_implement/README.md` | `3211c59c536a52e2ac5c76ce1b4842efe6c0584c38ab24a4be9b1ae776fa5fd5` |
| `docs/workflows/protocol.md` | `ee1749538322e2743920de9aea5c1a1158bee173ea38aa6785e15d84e3d33d26` |
| `docs/workflows/standalone.md` | `b8c1dd79117105e5108f1afcfe07953807d5b5b82b35e8876cb71c60a91f2cc8` |
