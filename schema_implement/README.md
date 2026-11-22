# Implement

把固定完整 Schema 落实成一个覆盖任务必要域和分支的 Python `Model` 入口，使用自有只读 coder 与正式 Agate SDK，在同设计累计调用余量内修复代码并全案例重验。不依赖 AKA、旧 KernelSchema runtime、Workflow 或 Extract runtime。

```sh
python3 schema_implement/implement_kernel.py --task T/task.md --eval-task T/eval --schema S --gpu L20D --out R
python3 schema_implement/implement_kernel.py --task T/task.md --eval-task T/eval --schema S --reference T/language.json --gpu L20D --out R
python3 schema_implement/implement_kernel.py --resume R --feedback F
python3 schema_implement/implement_kernel.py --resume R --recover-rejected-request
```

`--schema` 可以是目录或其中一个 YAML；始终固定完整目录、dispatch、from 链和 Template 有效来源。`--kernel` 可以是完整源码目录或自足 Python 文件；`--feedback` 可以重复。全新 `--out` 不得覆盖/包含输入或落入保护/产品目录。恢复只沿原输入、模型、目录和累计上限；已完且无新反馈直接返回原结果。

首建可重复 `--reference`，提供语言说明、环境记录等补充资料；它与候选修复 `--feedback` 分开快照，resume 不接受新增或替换 reference。资料须为自足普通文件，或保留必要本地文件拓扑的完整目录。初始化检查 JSON 路径字段、Markdown 链接和可解析的本地引用：缺失、绝对路径或逃出本次资料范围的依赖返回 `needs_input`，应提供使用相对链接的完整目录；不自动联网或抓取任意 Wiki。带有顶层 Schema/kernel 摘要绑定的 JSON 属于候选证据，不能通过 reference 混入当前候选验收。

资料的版本、来源与适用边界仍须由 Coder 核对；Task 和固定 Schema 始终权威。新资料不隐式补写 Contract、替换 pending Wiki 或取消必要缺口，已有 `needs_input` 不因增加材料而重开。

默认模型 `qwen3.8-max[1m]`、effort `max`、最多 6 次 coder、每次 coder 和每次远端等待分别 3600 秒；`--model/--effort/--max-calls/--timeout-s` 可在首建覆盖（超时须为正有限值）；resume 继续使用原运行配置。每次调用先计数，坏返回和超时也消耗次数。进度到 stderr，初始化后的 stdout 只有绝对 result.json 路径；退出码 `0/1/2` 表示 complete/incomplete/interrupted。参数错误不建立运行。

模型通过 native `claude` 的 Read/Glob/Grep 阅读全部 raw/effective 设计、DOC、固定判据和完整源码。它不能写文件或提交 GPU。host 接收完整 `files`、入口与落实说明；不采用 patch 或 shell 指令。保持当前 HOME、CLI 登录来源与 CLAUDE_CONFIG_DIR；关闭 hooks/settings 后，仅将已有 settings.json.env 的连接字段补到 CLI 环境，不将凭据写进产物。

运行需要 Python>=3.11、PyYAML、jsonschema；评测额外需要官方 `atrex-gateway-client`，本模块以发行版 **0.14.37** 核对和测试。发行渠道见 [官方安装文档](https://atrex-gateway.alibaba-inc.com/docs/skill)。不在 host 安装 torch、Triton、Bench 或 Ray。Agate 使用现有 `AGATE_AK/AGATE_SK` 或 `~/.atrex/config.json`；`KS_AGATE_CONFIG` 可指已有配置，`AGATE_URL/AGATE_PROFILE` 沿官方配置解析。不输出认证字段。

固定 eval 目录包含 `reference.py` 的 `Model(**init_kwargs).forward(**inputs)`、`input.py` 的 `_make_inputs(**input_kwargs)`、非空 `shapes.json`、所需 `metadata.json/roofline.json` 与 `eval.json`。eval.json 使用 SDK 名称：`operator,languages,requirements,harness,atrex_bench_version,mode,options,runner_overrides,deps_mode,lock_clocks`；显式固定 options 的 atol/rtol/num_correctness_cases/bench_iters/timeout_s，不借 SDK 默认放宽原判据。当前适配器不能表达的额外比较器文件/原生构建返回 needs_input。完整例子见 [vector_add](../schema_workflow/examples/vector_add/README.md)。

固定配置中的历史名称 `harness: atrex` 在 SDK 请求中映射为服务枚举 `atrex_bench`，也可直接使用 `atrex_bench`；固定 eval 文件保持原字节。服务校验失败保留字段位置、错误类型和经裁剪/脱敏的消息，不保存原始 payload、失败 input、上下文或认证信息。

产物只有 `inputs/`、`state.json`、`result.json`、`attempts/NN/`。源码固定后不再覆盖；终态反馈固定后不覆盖，修复进入新 attempt。result 仅含 `status,stop_reason,calls_used,schema,kernel,feedback`，路径相对 result。反馈中入口、源码和 Schema 的逐文件 SHA256、整体摘要、有效来源身份、落实说明、逐 case 判定/时延/单位/统计、实际环境及 request/response/job 在一处；所有反馈路径相对反馈文件。整体摘要为按键排序、紧凑 ASCII JSON 的 `{relative_path: file_sha256}` 映射的 SHA256。

反馈的 `materials` 只列固定 supplementary references 的实际快照路径、原位置、逐文件摘要和整体摘要，供后续 Optimize 沿原反馈读取相同正文。它不复制历史 repair feedback，也不声称模型已阅读或资料已成为本次候选的测量证据；`result.json` 字段不变。

自足文件按源码发送；真实多文件依赖打包，以官方 prepare_uploads/upload_file 和 `{archive,entry_point}` 提交。动作、固定请求和稳定幂等键先落盘。已知 job 仅查询该 ID；丢失提交回包沿官方幂等语义重发同一请求/键。短时 404、断连、未结清 cleanup、取消受理均不新建 job。默认不取消。SDK 0.14.37 没有旧上传签名 URL 刷新接口；上传进程丢失且原上传未确认完成时明确保留 reservation 和 interrupted，不另造 reservation 或把失败送 coder。

`--resume R --recover-rejected-request` 仅恢复已持久记录的 `rejected / 422 / request_validation_failed`，要求没有 job、上传 reservation 或未决动作，候选完整且 gaps 为空，所有固定证据与绑定一致。目前只允许 wire 的 `harness: atrex → atrex_bench` 适配差异；重建 body 不变或其他字段变化均拒绝。它在同目录新建 attempt，复制原 kernel/notes 并复用固定 Schema/eval/references，不调用 coder、不增加或清零累计次数。原 attempt 全部保留，新 key 对应修正 body；`state.request_recoveries` 和新 attempt 的 `recovery.json` 记录原拒绝、原 key/请求摘要及来源 attempt。旧拒绝反馈不混入新成功反馈的 Optimize 绑定。开关不能追加 feedback；已 complete 直接返回。若恢复后出现未决提交或已有 job，后续使用普通 `--resume R` 沿已保存的 key/ID 继续。

只有所有固定 case 均有比较器实测通过和有效中位时延，且没有已知必需设计/覆盖/代码缺口时 complete。真实基础设施失败为 evaluation_failed；编译/数值/代码落实失败才允许同设计修复。样本通过不构成全动态域证明，不另建静态证明器、reviewer、排名或性能搜索。

CPU 回归：`python3 -m pytest schema_implement/tools/tests -q`。Template 与所有输入只读；仅新候选源码由 host 写入。
