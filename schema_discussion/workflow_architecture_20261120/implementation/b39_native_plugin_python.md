# B39：标准工作区插件的 Python 入口兼容

2026-11-21 UTC。完成最小源码修复与 CPU 回归，**模型调用 0、GPU 调用 0**。普通 Python 可以加载由 runner venv 安装的插件锁，插件实际执行仍使用原锁中的 venv。没有启动新 campaign，没有修改 B35/B36 工作区、锁或历史结果；源码提交由主代理完成。

## 修改与边界

工作仓库 `/home/shaohanyong.shy/atrex-kernel-agent`，分支 `feat/schema-workflows-v1`。开始 HEAD 为 `621481a9b8685f4880e32a9ddce56f4e3d94cd07`；并行 B38 固定为 `ed63e97f3a4f872b55365f1120e15cda751e3803`。主代理完成窄 diff 审阅后，将本阶段三个文件精确固定为 **`ef8dadadd9867f298826d52ea159cf70bef93060`**；本代理未提交或推送。本阶段只修改下列三个源码/测试文件，未改 B38 文档、metadata、release.json 或 gitlink，未触碰 KernelSchema 算子提取。

| 文件 | 变化 | 当前 SHA256 |
| --- | --- | --- |
| `src/aka/resources/plugin_runtime/execution.py` | 解析 argv 可显式指定 `{python}` 替换值；默认仍为 `sys.executable` | `b96535f4e6b505bc0e7f7a0c438252373472b03f640a8bee68572bf563cfdc50` |
| `src/aka/resources/plugin_runtime/registry.py` | 整锁比较前，允许显式 `{python}` launcher 重新绑定到原锁解释器；整锁及挂载检查通过后才应用执行命令 | `11be5028816b4a3d081ecbabb7abd9c81913d7f877c2992adae426ca49b3e9d4` |
| `tests/test_plugin_python_lock.py` | 新增真实双 Python 进程的安装、发现、调用和拒绝回归 | `6f9b8957c33b15cec832a068e2ff4c16bed068cc75fe8e28da12c1c7d5d2c73e` |

精确修改清单、原文件 SHA 与检查命令见 [change-manifest.json](/tmp/schema-b39-native-plugin-python-20261121/change-manifest.json)，代码 diff 见 [tracked-code.diff](/tmp/schema-b39-native-plugin-python-20261121/tracked-code.diff)。新增测试原被现有 `.gitignore:19` 的 `tests/` 规则忽略；主代理已经精确 `git add -f tests/test_plugin_python_lock.py`，并包含在上述提交中。

修复入口在 [registry.py](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/resources/plugin_runtime/registry.py:289) 与 [execution.py](/home/shaohanyong.shy/atrex-kernel-agent/src/aka/resources/plugin_runtime/execution.py:17)。仅 manifest 的首个 command 项精确等于 `{python}` 时，才可使用原锁的绝对 launcher 路径。所有 `{python}` 参数使用同一锁定值；保留 venv 的原词法路径，不解析成系统解释器。插件 id/version/root/fingerprint、发现路径、命令其余参数、普通 PATH 命令和挂载仍按原规则比较。锁定解释器不存在时拒绝，不回退到当前 Python。

锁格式与 `api_version` 不变；不删除、不重写旧锁，也不忽略任意命令差异。完整匹配前不会修改 registry 的执行绑定。没有改动独立 retained Resources snapshot/checkpoint 的严格运行时绑定机制，也没有增加迁移或恢复平台。

## CPU 证据

最终 [插件回归](/home/shaohanyong.shy/atrex-kernel-agent/tests/test_plugin_python_lock.py) **10 passed in 6.27s**。使用 `venv.EnvBuilder(with_pip=False)` 建立真实第二入口，两个独立进程双向安装/加载，经原公开 CLI `list` 与 `call` 执行本地 CPU JSON 插件，验证实际 `sys.executable`、venv `sys.prefix`、含 `{python}` 的参数以及锁字节不变。另覆盖 adapter 内容、合法 schema、version、command、显式解释器声明变化、PATH 重绑定、锁定解释器缺失、剩余锁定 argv 不一致的拒绝。

与既有 `tests/test_schema_standard.py` 的联合检查在插件初版 9 项时为 **22 passed in 28.58s**；既有标准 Schema 13 项全部通过。随后为隔离 PATH 差异及增加剩余 argv 用例，只重跑最终插件 10 项。`git diff --check` 通过，没有全面重跑。命令与结果见 [checks.json](/tmp/schema-b39-native-plugin-python-20261121/checks.json)。

另用当前修复源码在 `/usr/local/bin/python` 下对**真实 B36 原 Agent 工作区**执行只读 `PluginRegistry.check_lock`：

- 检查前 snapshot 与原锁不等；检查后完整 snapshot 与原锁相等。
- `gpu-wiki.query` 绑定仍为 `/tmp/schema-b18-model-budget-usage-20261121/venv/bin/python`。
- 原锁 SHA256 前后均为 `afde8bbd424af923931284d9f1fecb9b104c173ebabcd1df3abf0e82d617b954`。
- 未调用 Wiki 插件或任何模型/GPU。

精确路径与回执见 [existing-workspace-check.json](/tmp/schema-b39-native-plugin-python-20261121/existing-workspace-check.json)。这验证的是插件锁兼容，不能充作候选 kernel、GPU 评测或下一次模型完成率证据。

## 下一次原 Light 的最小调用建议

B35 的 600 秒和 B36 的 900 秒都没有产出 candidate。B36 公开工具记录为 13 Bash、15 Read、0 Write/Edit，实际停留在目录/文档/环境/Wiki 准备；其中缺 scratch 目录已自行恢复，插件解释器错误已证实。仅凭这些事实，不能认定延长 timeout 就能解决，也不能把本次 CPU 修复当作下一次必成功。

建议保留原 Schema `root.yaml` 原文，`docs=()`，继续原 `aka.workflow.schema.standard.implement`、原 Light Agent、原单批 full Evaluate 与原 native ABBA。此次简单 vector-add 不需要 Wiki 调研，可在**本次调用的现有 composition patch** 中加下列资源配置：

```json
{
  "id": "resources",
  "config": {
    "plugins": [],
    "bundles": ["measurement-session", "research-records"]
  }
}
```

原 Light 默认 `workflow.wiki=false`、`research_records=false`，但默认 resources 仍选中 `plugins/gpu-wiki`，其资料含 episode 开头必须查询的说明。显式取消这个非必要插件可减少工具准备与歧义；不用删除原默认能力或增加开关。保留的两个 bundle 继续提供 `gpu-measurement`、`runtime-records` 与 `tools/sandbox.py`。保留原 episode 中 research recording disabled 的终端报告契约，不额外开展 Direction/Experiment 准备。

原 `notes` 可简化成：接口与形状约束、只使用给定工作区的公开资料与受保护 Python/原 sandbox 工具、读取 `.claude/skills/gpu-measurement/SKILL.md` 一次、立即实现、一次覆盖三个 shape 的 full Evaluate、按原报告契约提交真实测量。继续保留用户现有的访问范围和费用限制，不附带全部 Schema 手册、重复工具发现或 host 环境排障任务。不要为此增加自定义 coder、预算、资格门槛或验收框架。

完整示例 [minimal-light.patch.json](/tmp/schema-b39-native-plugin-python-20261121/minimal-light.patch.json) 直接保留 B36 的原 agent900s/episode1100s、shape_batch_size3、native-abba、measurement_repetitions1，并只追加上述 resources 配置。正常公开接口的零调用 `prepare_only` 已通过，使用原 B36 Schema/Task 输入；与 B36 composition 比较仅 `resources` row 改变，Schema context 完全一致，目标 workspace 没有创建。回执和精确调用 options 在 [minimal-light-prepare.json](/tmp/schema-b39-native-plugin-python-20261121/minimal-light-prepare.json)。仍为 max_iters1、handoff_resumes0、max_stall1、原 token admission600000、setup_timeout600、verify_repeats2/per-run120；没有提高费用或增加调用。

可复制的 CPU 预检命令：

```bash
cd /home/shaohanyong.shy/atrex-kernel-agent
PYTHONPATH=src /tmp/schema-b18-model-budget-usage-20261121/venv/bin/python - <<'PY'
import json
from pathlib import Path
from aka.workflow.schema.standard import implement
saved = Path('/tmp/schema-b39-native-plugin-python-20261121/minimal-light-prepare.json')
options = json.loads(saved.read_text())['options']
print(json.dumps(implement(**options, prepare_only=True), indent=2))
PY
```

下一次真实运行仍由主代理把同一普通 `implement` 调用放回现有原父账执行方式，保留原费用归属和用户限额，选新 workspace；本阶段不启动、不新增账本或运行包装框架，也不修改旧 campaign 以冒充恢复。Light 目前的 episode effort 为原固定 `max`，没有为改 effort 增加源码开关。

全阶段仅访问指定 AKA、报告和精确证据目录；未读取、枚举、运行或修改任何 `AKA_experiment`/archive 变体，未从 home 递归搜索。附带输出只有本报告和 `/tmp/schema-b39-native-plugin-python-20261121` 下的 CPU 证据。
