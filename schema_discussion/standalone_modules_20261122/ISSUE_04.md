# Issue 04：正文 Grep 阅读证据与实测等待时间

`vector_add-004` 在原 host 下已终结为 `incomplete / write_failed`，Design 用完 3 次调用，`schema=null`、Workflow `last_usable=null`，未进入 GPU 候选评测。修复不修改旧运行、原稿或结论，也不把已经写出的 Schema 当作已验收交付。

## 真实观察

001–003 的写作在 900 秒处被截断；003 只留下真实模型的部分 root 草稿。004 经正常 Workflow 入口，以 `--reference` 提供明确未验收的 003 历史稿和真实环境记录，显式使用 `--timeout-s 3600`，模型与 effort 仍是默认 `qwen3.8-max[1m] / max`。

004 的公开 `design/drafts/NN.json` 记录如下，耗时使用 host `elapsed_s`，不混同 native `duration_ms`：

| 调用 | host 耗时 | native 终态与产物 | 原 host 结果 |
| --- | ---: | --- | --- |
| 01 select | 156.429 秒 | success，returncode 0 | 选材成功 |
| 02 write | 1775.020 秒 | success，3 次完成 Write，交出 root.yaml、multi_tile.yaml、dispatch.md；本次 Read 了 DOC1–6 | 报出 13 处 text 长度问题，进入具体修稿 |
| 03 repair | 508.348 秒 | success，10 次完成 Edit，再次交出完整三个文件；完成两次 DOC 正文 Grep | 被只认 Read 的阅读 gate 拒绝，未进入本次稿的正式收取与规范化 |

03 对 `5.design.tile_pipeline.md` 使用 `meaning|describe`，对 `1.contract.algorithm_requirement.md` 使用 `range|meaning`；两次均为 `output_mode="content"`、`-n=true`、`status="completed"`。原诊断是 `Writer must actually Read the relevant DOC1–6 material before deciding`。公开记录证明正文模式请求已完成，固定 DOC 字节中也存在对应匹配项；旧解析器未保存成功工具返回正文或命中元数据，因此不能声称从旧 JSON 直接看到了返回正文，也不能事后补造新 content_hits。

主代理对 `004/design/drafts/03/root.yaml` 执行正式 Template `--read-only`：structure、text budgets、from、relations、dispatch 均通过；2 份 YAML 仍有规范化预览，故 CLI 返回 1，原稿未改。这表明模型修稿已消除已见问题；它不等同于原运行已被 host 接受，更不证明数学正确性、实现或性能。

## 根因与修复

D/O 各自本地 runner 的 gate 仅接受本次成功 Read 的 DOC1–6 精确路径，忽略了同样真正返回相关正文的 Grep。修复保留原 Read 规则，并从公开成功 `tool_result` 中提取最小 `content_hits=[{path, lines}]`，不保存整段正文或思维内容，不引入共享根包或资格账本。

只有显式 `output_mode="content"`、成功返回、含正行号且正文非空的匹配行能形成 Grep 证据。上下文行、文件列表、计数、无匹配、空白、失败或仅请求尚无返回均不能充当阅读。单文件搜索可使用输入路径定位；目录搜索必须由返回行明确指出实际文档。绝对路径、cwd 相对路径及搜索目录相对路径须最终落到当前固定快照的 DOC1–6，且位于本次请求的范围内；同名其他文档和 DOC0 不满足此要求。未带可核实行号的正文保守地不计，prompt 明示使用 `-n=true` 或 Read。

每次冷调用独立提供证据，包括具体修稿；不把 02 的 Read 自动归给 03。D/O 的提示和 README 同步说明 Read 或正文 Grep，不再要求 Read 工具名本身的仪式。材料与 Schema 的其他检查、有限调用额度和候选保存边界不变。

1775.020 秒的完整真实交稿已经超出原 900 秒默认值。四入口统一将新运行的 `--timeout-s` 默认改为 3600 秒：D/O argparse 默认、Implement 首建配置、Workflow 首建配置。仍接受正有限覆盖值，resume 沿用原配置；Qwen/max、Design 3 次、Optimize 2 次、Implement 6 次以及 Workflow 版本与调用额度不变。固定 `eval.options.timeout_s` 不变，没有逐阶段时间矩阵。3600 秒是依据本次实际耗时选定的统一等待值，不保证所有任务都能在此时间内完成。

## CPU 验证与后续真实边界

使用既有 `/home/shaohanyong.shy/.cache/kernelschema-standalone-venv/bin/python`：

```sh
python -m pytest -q schema_design/tools/tests schema_optimize/tools/tests schema_implement/tools/tests schema_workflow/tools/tests
# 97 passed, 70 subtests passed in 55.30s

# 完成集合后，再补正文含文件定位符的歧义边界，只重跑对应文件：
python -m pytest -q schema_design/tools/tests/test_doc_reads.py
# 4 passed, 64 subtests passed in 1.25s
```

新增测试从公开工具事件经过 `parse_stream` 到两个本地 runner 的真实 `Run.call` gate，覆盖成功 Read、Grep 正文、目录归属、空匹配、列表、计数、权限失败、缺少返回、不保留正文/思维，以及修稿不继承上次阅读。另用模型/SDK 均被替换的 CPU 配置检查，确认四入口默认 3600、17.5 秒覆盖、0/负数/NaN/Inf 拒绝、原 900 秒 resume 保持且不能覆盖，以及 Qwen/max/调用额度不变。`git diff --check` 通过。

本修复未新启动真实模型或 GPU，未访问原生大捕获或思维材料，未修改旧 run、保护资料或固定 Task/eval。共享 EXECUTION 与提交由主代理处理。

修复提交后，正常后续入口应创建全新 Workflow run，使用默认 3600 秒；通过 `--reference` 提供真正模型完成的 `004/design/drafts/03` 全稿、明确其未验收状态的说明，以及原真实 environment 记录。由正常 Design 重新取得本次阅读证据、交稿并经过 host 的规范化与校验，再继续 Implement/Optimize。旧 004 不重标、不恢复成成功；新的真实闭环仍须以新运行产物和固定全案例测量为准。
