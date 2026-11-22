# B23：standalone 真实链预检与有限预算入口

2026-11-21 UTC。本代理先按根安排只做零调用预检；发现原公开 Demo 的固定 1,200,000 ms 请求及默认写作最低窗口无法表达本轮两次 600 秒写作调用后，根授权本代理作为唯一 driver writer 修复最小入口。本项尚未运行真实业务链，**新增模型调用 0、GPU jobs 0、真实 owner 注册/预约/计时 0**。没有建立或提高账户，没有提交/推送，没有派生代理，没有访问禁止的实验仓。

生产修改只在 `/home/shaohanyong.shy/KernelSchema/.worktrees/schema-workflows-v1` 的 driver、CLI、对应 standalone tests、standalone 文档和 WRITING 文档。没有修改 evaluation、预算实现、runtime、原 Task 判据或原 Schema。原 main 只写本报告。

## 固定来源与真实输入

起点为 KS `c17c15d8842b8ed13dc34fa0530cf112a9d4ce82`。独立源码快照保存在 `/tmp/schema-b23-standalone-real-loop-20261121/fixed-ks`；对实际 KS、row-softmax、Rules/docs/Template 相关 **168 个 Git blob** 逐字核对，manifest SHA 为 `a24d0f1d46e0e2cde969955954412a7d7bac16c7d7e73931512d15afa36abf77`。

零调用预检脚本及结果为同目录 `source_preflight.py`、`source-preflight.json`。脚本使用 B18 现有 Python 3.13 解释器并显式把本项固定 c17 源码置于 import 首位；实际 api/budget/writing/runtime/evaluation/reference 模块、`resources.checkout_root()` 和 Template `schema.py` 均核到本项快照，不把 B18 的旧 77ad editable 安装冒称新版本。没有安装或下载 Python/依赖。B18 环境有 PyYAML 6.0.3、jsonschema 4.26.0，但没有官方 Agate SDK；B22 提供的既有 GPU 解释器为 `.worktrees/.tooling/k1-evaluation/bin/python`，SDK 0.14.37，后续若合法需要时可复用，不由本项安装。

本项只读核验 B22 的真正 c17 `Demo.create` 准备产物，不用旧 B16 Task 替换新 worker：

| 对象 | SHA-256 |
| --- | --- |
| Task | `f051f7152b3149b1d16505b68f143ceca2154ade2d365841042f6e97b5448652` |
| Evaluator | `5f82fb581ffd8cba224eb4bda68155c3eba20816946fc14b657c4fa20340a69e` |
| B22 target-bound R0 | `3239691d521387feb1f6b3b6bf3c5bd0316eb51c1c8c8f8441b73b9807d6fbff` |
| 原 L20D target | `d4b9c90c350b4db10664ac9ee66ec118c852a03f5c9ffd955204ea89e31ddb0f` |
| exact 2.00 USD / 600000 ms Claude profile | `3e1766d7fad58da4e76b77c28a296b0adcb225c04e7e56704dac45a848efc530` |
| 原 public parent | `85f19bfe92d717c94950442d3fbab16e63bccf52a0f4cb8594a71c6fd02ca3a7` |

准备 handoff 为 `/tmp/schema-b22-gpu-qualification-20261121-rnzs0ew0/preparation-handoff.json`，bundle 为同根 `standalone-demo`。五个实际请求、Task/evaluator/Schema/Scope/Rules/Materials/profile/parent 的 Ref 字节全部校验；target 与 Task.environment 相同；Rules 保留实际 docs/Template；原 `workflow_objective.md` 字节没有替换成 transport objective；design 请求没有输入 Schema，Task 派生 locked Contract 可合法生成。静态 scope pass 只表明四形状绑定完整，不构成 GPU 或 fidelity 资格。

B21 native coder/reviewer 成功仅复用其 exact CLI profile 和现有运输事实，不能代签 KS 的 planner、RoleReceipt、独立语义/fidelity review 或业务 Task。后续原 72+2 状态调用、x/out/storage/guards、FP64 golden、完整 forward 及 80 AB/BA batch 判据均不在本项修改范围。

## 实际发现与修复

原 c17 `_limits()` 对 design/implement/optimize/verify 均固定 `active_ms=1200000`；`Demo.create` 也不能设置 `writing_minimum_active_ms`、reference `minimum_active_ms` 或 `writing_finish_floor`。只给 ClaudeProfile 600 秒不能替代两个实际最低窗口；默认 planner1ms/reviewer1000ms 不满足本轮预留要求。

新增四个可选 Python 参数，并在 `prepare` 增加四个对应 JSON 文件选项：

| 参数 | 冻结位置与作用 |
| --- | --- |
| `request_limits` | 必须恰含 design/implement/verify/optimize 四项完整原预算结构；解析后冻结在 config，五初始 Q 及每次后续 `_request` 均读取它。fixed/partial/code action 用 implement；verify-only 用 verify。 |
| `writing_minimum_active_ms` | 正整数 planner/reviewer 映射，冻结在原 writing profile。 |
| `reference_minimum_active_ms` | 正整数 coder/evaluator/reviewer 映射，冻结为原 reference profile 的 `minimum_active_ms`；真实 qualification 注入保留该参数。 |
| `writing_finish_floor` | 原完整 `budget.cost()` 向量，非负整数且 model_calls 等于角色 calls 之和，冻结在 writing profile。 |

格式、未知/遗漏字段、bool/非正 minimum、不安全整数、floor 总数不一致等在创建输出前拒绝。调用者随后修改输入 dict 不会改变已冻结配置。默认仍保持原请求限额及 backend minimum/floor 默认行为。全部实际准入仍由原 OwnerAccount 同时检查 parent/request/native/截止与余量；没有假 operation、预扣费用、第二账本或续额入口。

对本轮长 writing，选 planner/reviewer 各 `600000`，无额外 finish floor 时 request 至少 `1250000`，给两次调用及宿主处理留空间。若要预留后续 fixed 请求，写作请求自身的 roles/jobs/active 也必须容纳完整 finish vector；仅增 active 不足。reference coder/reviewer 的长窗口同理需要显式设置，另留实际 evaluator 容量，不能用默认 1000ms 宣称足够。

## CPU 验证

使用既有 `.worktrees/.tooling/k2-cpu/bin/python`，设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`；全部运输明确为 synthetic。本项不运行全库。

第一组 `-k explicit_budgets` 为 **12 passed、1 failed / 117.72s**：两个完整链、参数传播/恢复/六个合成资格、十类提前拒绝通过。唯一失败是新 CLI 测试错误地要求 authority 整份字节不变；原 CLI 会调用 summary/fences，正常刷新 idle clock 基准。修正为与原 CLI 测试相同的 descriptor/operations/requests/paused/active_ms/active_ns 不变断言，没有修改生产预算行为、原账户或费用。

随后 `-k 'explicit_budgets_cli or explicit_writing_minima or driver_two_rounds_real_api or five_requests_and_explicit_initial_source'` 为 **7 passed、39 deselected / 87.12s**，覆盖修正 CLI、请求/parent 过紧各自零调用、默认完整链及 fixed/partial/verify 默认入口。最终相关 **19 个不同定点全部通过**；没有无故重跑已通过的昂贵链。三个 Python 文件 AST 和 `git diff --check` 通过。

两个完整新参数测试分别实际经过 synthetic design→implement→optimize→implement 与 keep/verify，沿同一个已有 synthetic parent，六个原 synthetic qualifier jobs 后在 writing 发布点中断并恢复。初始五 Q 与后续 Q 的各阶段限额准确，qualification 注入不丢 reference minimum，planner/reviewer minimum 和 finish floor 保留，调用者改传入 dict 无效；原历史 operations 不变、没有新 owner、terminal replay 不追加调用。没有把这些 CPU 结果当作真实 writing 或 GPU 通过。

## 阶段预算与当前合法分支

资格完成后的成功最小完整 design→implement→optimize→implement 需要 **8 个模型调用、2 个业务 jobs**，角色为 planner2/coder2/reviewer4；若 follow-up 是 verify，则为 **7 models、2 jobs**。原每个 code 请求允许两次 coder/evaluate/reviewer时，前者最坏 **12 models、4 jobs**，后者最坏 **9 models、3 jobs**。fixed 起步分别少初始 writing 的 planner1/reviewer1。optimize keep/none 是合法终点，不强迫改 R 或附加 action。以上是阶段调用上限，不是必付费用，也不把旧 R 通过当新 R 通过。

根开始协调本补丁时，B22 原 probe 已发出；本项没有运行或改写它。B22 随后报告实际 L20D job `dv_80d527ccf029` 在 fresh `/proc` mount 被拒（inner exit1、mount exit32），尚未到 stage bind/torch；其余五 job 未投。原 suite 保持不合格，不改判据，不换通道。B22 报告已结算、无 active/fence；其当时快照为 parent 累计 model3/jobs3/active388656ms，剩 model21/jobs13/active10411344ms。该值只作注明时间的交接事实，后续收费前必须重新检查原 owner，不能用本报告快照计算余额。

GPU 未合格时，唯一可独立完成的真实业务分支是纯 design writing：新 Demo(start=design、原 parent、exact profile、充足已冻结写作预算) 的原 `run()` 会先调用真正 `api.design`，最多 planner1/reviewer1、GPU0；若 design completed，则原控制流进入 implement admission，由于 GPU 未合格零 coder/零 job blocked 并停止。它保留真实 writing 发布，**不构成完整 kernel/standalone 链通过**。当前 driver 不允许在已有 history 后追加 prepare_agate；不能承诺在此同 Demo 后补资格继续，也不能改其冻结 state/profile。直接公开 `api.design` 也是合法单独 writing，但不能代签原 driver 完整链。

本项尚未创建任何新的真实 Demo、register/begin、触发模型/GPU或读取实际 owner 计时。根固定新 driver 并决定下一阶段后，才可准备新的同 parent 请求。旧 c17 Demo/其已付费 probe 必须保留，用原固定安装追溯，不将新 driver 绑定到旧 anchor，不增加 attach_qualification 平台。

## 封存与可执行准备

五文件 freeze 为 `/tmp/schema-b23-standalone-real-loop-20261121/source-freeze.json`，只含本项归属的 `owned.patch` SHA 为 `0bf96aa4beb106583431fb95c306ada0dce39c3596d501f900f985ab25ae05db`；driver SHA 为 `f4e117cc9ad46271ab2f89dcb865fbf0f89a8e0182dcddb39551881bbf92d863`。根后续提交的 Git SHA 再绑定实际业务安装，本项不自行提交。

自己的 `inputs/` 已从 B22 公开 Ref 逐字复制 target-bound R0、dispatch、target、remote、原 parent Ref、exact Claude profile，并记录 `origin.json`。两个尚未准入的预算提案分别为 `pure-design.json` 和 `full-one-episode.json`：前者每次 writing 1,250,000ms、600秒+600秒、finish floor零；后者每次 writing 2,700,000ms，另保留后续 coder1/reviewer1/job1/1,320,000ms 的原 finish vector。两者 reference 均为 coder600000/evaluator120000/reviewer600000，单次 code 请求只容许一轮（1,500,000ms、coder1/reviewer1/job1），verify 请求800000ms/reviewer1/job1。后者是未来资格通过后的有限提案，不是当前 GPU 可用承诺；额外实现修复必须由根事先选择不同的新请求预算，不能改已准入请求。

`prepare_demo.py --help` 已通过。该脚本严格核 B23 driver SHA，通过公开 `Demo.create` 冻结新目录，仅复用原 parent，核准备前后 authority 字节相同，**没有调用 run/qualify/API admission**。目前只执行过复制输入脚本和 `--help`，没有实际执行此准备命令。根固定代码及决定分支后可执行：

```sh
/home/shaohanyong.shy/KernelSchema/.worktrees/.tooling/k1-evaluation/bin/python -I -B \
  /tmp/schema-b23-standalone-real-loop-20261121/prepare_demo.py \
  --source-root /absolute/new-fixed-B23-source \
  --out /tmp/schema-b23-standalone-real-loop-20261121/new-design-demo \
  --proposal pure-design
```

必须将示例 source-root 换成根已固定的新源码快照，旧 c17 快照因 driver SHA 不符会拒绝。该准备命令自身不构成真实 writing 执行授权；收费 slot 与具体下一步仍由根协调。
