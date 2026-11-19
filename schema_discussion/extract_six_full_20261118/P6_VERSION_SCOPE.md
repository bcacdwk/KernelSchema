# P6 版本 scope 拒收与恢复

2026-11-19。Sparse run `8f0844f9-f279-4781-a19d-0216b7fc4ef5` 的 P6 reviewer attempt `2f017dd3-b509-43f1-89ca-b8ebde2376bb` 被拒原因是 `P6 field exceeds the allowed node scope: META_DATA.Version`。

真实提案是 `revise`：要求把 `dispatch.md` 的计划表述改为最终目录并订正 CM、IS、STAGES 记录；`tcgen05_packed.yaml` 的 SOFTMAX/LOADER 两条 `store.zero_rows.describe` 改为各角色承担跨步份额，同次可选精简 `REALIZATION.Prefer`，随内容修改递增版本。其 `revision_scope` 除这些字段外显式列了 `META_DATA.Version`；未请求写 root。此处只核对权限和恢复条件，不代替独立语义审阅，也未将被拒提案转成批准。

当前及本批 frozen 的 `validate_revision_scope` 仅接受目录和指定非 root 内容字段；`_erase_fields` 已在获准文件的内容范围比对中忽略 Version，`_versions` 仍要求内容修改递增正整数版本。显式 scope 用来授权内容修改，版本是随该修改承担的责任。无需新增 Version 独立修改入口：只在 [P6 阶段提示](../../schema_extract/prompt/phases/06_final.md) 补一句，说明 Version 随已授权非 root 内容更新、属于隐含写权限、不单列于 `revision_scope`，由 Runner 校验。未修改 schema、校验代码或本批 frozen。

恢复前已核对原生单一 success、exit/cli-exit 0、schema/nonce 有效、session `98b96fbc-af7d-4563-8201-0dadef08a946` ready、termination confirmed 且无 survivors。登记 Runner/CLI 及后代 PID 均已退出，公开 inspect 显示 runner lock free、alive false；active/unsettled/unconfirmed 均为空。P6 已用 calls=2、revisions=0、invalid_receipts=2，max_calls=32，call_additions 为空。

02:33:07 UTC 通过公开 whole-run `resume` 恢复同一固定 run，命令及 PID/log 已追加 [RECOVERIES.jsonl](RECOVERIES.jsonl)。确认新 attempt `14a75cae-ee8c-4a19-b529-1af1f3bdaf0f` 为 active P6 reviewer，精确沿用原 session 和 checkpoint `71ec255351ca0a11d68a38db310dfa981ba5f04967fe0b97fa7b13dbfd7d0425`，P6 calls 自然增至 3，无额度追加；随后停止控制动作。回执由原 reviewer 按冻结规则重交，未手改回执、产物、状态、批准或快照。启动恢复不等于 P6 已通过。

定向内存检查通过：授权非 root 内容可连同隐含 Version 递增；显式 Version 仍拒绝；root 和 reviewer 保持只读；其他 metadata 仍受保护；改内容不增版本仍拒绝。当前与 frozen 的五个权限/版本函数 AST 一致，`git diff --check` 通过。仅补提示与本报告，未启动额外模型，未提交或推送。
