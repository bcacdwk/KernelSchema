"""Close out D only after affected regression checks have passed."""
from pathlib import Path
import ast,hashlib,json,re,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'schema_discussion/main_integration_20261122';OUT=BASE/'round2_d';P='schema_discussion/main_integration_20261122/'
def load(p):return json.loads((ROOT/p).read_text())
def ref(p):return {'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()}
initial=load(P+'round2_d/workflow_initial_verification.json');affected=load(P+'round2_d/workflow_affected_verification.json')
final=load(P+'round2_d/workflow_final_case_verification.json')
assert affected['exit_code']==1 and affected['input_bytes_unchanged']
assert final['status']=='passed' and final['exit_code']==0 and final['input_unchanged']
items=[x for x in (OUT/'collection.log').read_text().splitlines() if '::' in x]
rerun={x.get('classname').replace('.','/')+'.py::'+x.get('name') for x in ET.parse(OUT/'workflow_affected.xml').iter('testcase')}
expected={x for x in items if x.startswith(('tests/workflows/test_write_only.py::','tests/workflows/test_standalone.py::')) or '::test_strict_rejects_old_transitive_template_policy_without_rewriting_rules' in x}
assert rerun==expected and len(rerun)==70 and len(items)==187
failed={x.split(' ',2)[1] for x in (OUT/'workflow_initial.log').read_text().splitlines() if x.startswith('FAILED ')}
assert len(failed)==33 and failed<=rerun
for p,v in initial['inputs'].items():
 if p not in ['tests/workflows/test_writing.py','tests/workflows/test_standalone.py']:assert ref(p)['sha256']==v,p
new=(ROOT/'tests/workflows/test_writing.py').read_text();old=new.replace('rules["files"] = sorted((old_source if ref == source else ref for ref in rules["files"]), key=lambda ref: ref["path"])','rules["files"] = [old_source if ref == source else ref for ref in rules["files"]]')
assert hashlib.sha256(old.encode()).hexdigest()==initial['inputs']['tests/workflows/test_writing.py']
a,b=ast.parse(old),ast.parse(new);target='test_strict_rejects_old_transitive_template_policy_without_rewriting_rules'
a.body=[n for n in a.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!=target];b.body=[n for n in b.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!=target];assert ast.dump(a)==ast.dump(b)
standalone=(ROOT/'tests/workflows/test_standalone.py').read_text()
prior=standalone.replace('args = ["prepare", "--writing-mode", "strict", "--out", str(tmp_path / "demo"), "--parent-ref", str(ref_file),','args = ["prepare", "--out", str(tmp_path / "demo"), "--parent-ref", str(ref_file),')
assert hashlib.sha256(prior.encode()).hexdigest()==initial['inputs']['tests/workflows/test_standalone.py']==affected['inputs']['tests/workflows/test_standalone.py']
for path,value in affected['inputs'].items():
 if path!='tests/workflows/test_standalone.py':assert ref(path)['sha256']==value,path
assert ref('tests/workflows/test_standalone.py')['sha256']==final['test_input']['sha256']
summary={'format':'kernelschema.integration.workflow_closeout.v1','status':'passed','distinct_cases_verified':187,'initial_run':{'executed':187,'passed':154,'failed':33,'evidence':ref(P+'round2_d/workflow_initial_verification.json'),'log':ref(P+'round2_d/workflow_initial.log'),'log_publication':'local_only; finite failures summarized in compatibility_correction.json'},'affected_run':{'executed':70,'passed':69,'failed':1,'evidence':ref(P+'round2_d/workflow_affected_verification.json')},'final_case':{'executed':1,'passed':1,'failed':0,'evidence':ref(P+'round2_d/workflow_final_case_verification.json'),'correction':'Explicit strict mode for the historical reference-profile CLI budget fixture; other standalone bytes identical after reversing only this argument insertion.'},'reused_unaffected_passes':{'count':117,'cases':sorted(set(items)-rerun),'validation':'All recorded product/rule/test bytes unchanged except one new strict-drift test function. Reconstructing its prior unsorted statement matches the original test module SHA; AST excluding this function is identical. Public materials.json is only read by Demo creation in write_only/standalone, both rerun entirely. API, iterations, Template bridge and remaining writing tests do not read that material catalog.'},'not_double_counted':['Initial passes also included in the 70-case rerun','Three diagnostic failure reproductions','Historical study tests'],'environment':ref(P+'round2_d/environment.json'),'compatibility_change':ref(P+'round2_d/compatibility_correction.json'),'guarantees_checked':['main private Template namespace and effective root/from resolver','default write_only keeps raw text/diagnostics and single writer without implicit review/effect/GPU gates','strict 300/2 policy projection and actual complete frozen Rules; old transitive policy rejected before calls','Extract CLI help, invalid arguments and exit code passthrough','original standard AKA delegation mocked and lazy; no receipt/current gate','iterate per-version full campaigns, keep stop, failed feedback, unsettled/error stop and no replay'],'limitations':['CPU/fake backend only; no AKA tree/model/GPU/real Extract/monitor access','187 is the union of applicable cases, not a claim that the initial entire invocation passed']}
(OUT/'workflow_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
subprocess.run(['/opt/venv/python3.13.12/bin/python','-B',str(OUT/'verify_preservation.py')],check=True,cwd=ROOT)
subprocess.run(['/opt/venv/python3.13.12/bin/python','-B',str(OUT/'build_coverage.py')],check=True,cwd=ROOT)
(BASE/'ROUND2_D.md').write_text('''# 第二轮 D：独立复核、兼容与公开收口

D 的必需项已完成。W2 有限独立复核和 current 终检通过；workflow 187 个不同定向用例均已有对最终适用字节有效的通过证据。精确 stage 清单由 HISTORY_PUBLICATION 与本地 round2_d/stage_allowlist.json 给出，Git 提交和正常推送交根代理执行。

[W2_REVIEW](W2_REVIEW.json) 固定 SHA `af96e504979f3a98cd975f9f82922ad088821ad4a0c13ed0e217ea8d45679255`，只绑定 subject `4936e08c5611c703497c6e53a244ff4e3cf746f993ebff62dce57c60752a2242`、来源与 checker，不包含后来生成的 current 摘要。复核覆盖九 YAML、两 dispatch、19 YAML/9 dispatch 完整目录、五个固定 ACCEPTANCE 的九输出/十有效节点和真实隐式 root/显式 from 闭包。两处发现由 C 同轮更正：GDN 显写缺 CTA 聚合风险；mixed Sparse 区分缺原题回执和题目哈希未确认。受影响检查、subject 和独立证据刷新后才允许生成 current。

[九 current 与索引终检](round2_d/current_verification.json) 逐项确认 files/Version/Source/源 SHA、依赖、实际规则/有限审阅及原 receipt。九个实际目录共 27 个 bridge 断言通过：旧 receipt 拒新字节，curated+receipt=None 接受，current 作为 receipt 拒绝。原 receipt/DELIVERY_INDEX 和其他 24 YAML 未改；新 current 不承担 P6 批准。

[workflow 验证](round2_d/workflow_verification.json) 使用指定 pytest 环境、显式 main PYTHONPATH，Template 私有命名空间来源实测在 main，PATH 无真实 claude/codex。初次六模块 187 例为 154 pass/33 fail：32 例受示例 DOC1/4/5 旧材料 pin 阻塞，一例是 D 新增 Rules 反例未排序 refs。根在 [OWNERSHIP_CHANGES](OWNERSHIP_CHANGES.json) 精确分配 materials.json 后，只更新三条 size/SHA，并修该合成 fixture。70 个受影响用例先有 69 通过，余一原 strict CLI 预算夹具补上显式 --writing-mode strict 后单例通过（默认 write_only 没有该 reference profile，不改产品默认行为）。117 个不受影响通过结果按文件摘要/函数 AST 和依赖核对复用；不把重跑计为额外独立覆盖。初始完整失败日志留本地，公开修正摘要保留失败原因、原日志 SHA 和真正的后续通过。

没有改 kernelschema 的 parser、writer、bridge、api 或 iterations。默认 write_only 的单次 writer/原文保留，无隐式 reviewer/效果/GPU gate；strict 新冻结规则的真实 BUDGETS=300/2、旧规则漂移拒绝；标准 AKA lazy mock 委托与 iterate 的 keep/失败反馈/未结算停止/不重放均保持。文档材料 pin 是新版本公共输入更新，不迁移旧 Rules 或账户。

[导航核验](round2_d/navigation_verification.json) 为 18 项/48 YAML，原六行导航逐字保留；本批十二项区分九 current 和三原 receipt。[保护核验](round2_d/preservation.json) 包括 695 原件、277 允许保护文件、12 state/12 receipt/supervisor/AGENTS；42 archive 元信息先排除，不访问目标。旧 57 文件已从原基线提交逐 SHA 还原核验，workflow 37 个自然提交仍可达。

[实施覆盖](IMPLEMENTATION_COVERAGE.json) 收齐 26 发现、24 候选/104 出现、20 recap、11 既有修正、16 删减和 FC01/02 的实际路径、测试或保全证据。五个旧候选全文审阅通过；其余 690 原件保持 local-only，少量有限来源摘读与未内容审阅的捕获分别标明。主入口仅导航公开材料；fresh clone 的本地历史、原 runtime、/tmp 环境和禁 archive 来源边界见 [公开历史](PUBLIC_HISTORY.md)。

初始文件定位命令有一次范围偏宽的目录级探测可能，已在 [过程边界](round2_d/process_boundaries.json) 如实记录；未打开 archive 正文，archive 不构成验证来源。

所有 fixture 使用新临时目录；未运行模型/GPU/真实 Extract/monitor，未改 AKA、固定来源、旧 run/账/receipt 或 AGENTS，未做 Git 写操作。masked 4616 索引组合是旧证据复用，不是新测试；未证明全 33 稿、全域数值、编译器寄存器分配、来源风险消除或真实模型成本收益。没有本轮必需项遗留。
''')
(BASE/'FINAL_REPORT.md').write_text('''# main 整合与修订交付

授权的两轮 A/B、C/D 实施和独立验证已完成，产品及已审公开材料可由根代理精准提交并正常推送 main。原审计 FINAL_PLAN 的“待实施”仍保留其历史含义；本报告及 [IMPLEMENTATION_COVERAGE](IMPLEMENTATION_COVERAGE.json) 给出实际实施结果。最终 Git/远端 SHA 由根的交付记录绑定，不在自身提交中虚构自己的 commit SHA。

原十二项 57 文件已在 `41b7db0dd636f01a1d8bc490bc159c2d65b7b034` 保留批准字节；`6c3859e58cbc25e4094590c5e50f39ffcdb773a3` 正常合入 workflow 的 37 个自然提交。W1 `1a221e4` 修数学身份、P2 候选与控制提交事实；W3 `d444101` 统一 handoff/证据义务、安全完整诊断和普通补正的既有预算；W4 `f46b14a` 区分缓存/样本、当前/历史观察与 clock 代际。W2 九 YAML 各增一版、两 dispatch 更正来源边界，加入有限规则澄清与九 current；所有产品和覆盖文件由包含它们的 Git 提交绑定。

| 验证对象 | 本轮实际结果 | 证据 |
| --- | --- | --- |
| A：W1/W3 最终共同树 | 186 个不同 CPU 测试通过 | [ROUND1_A](ROUND1_A.md) |
| B：W1/W4 最终共同树 | 46 个 CPU 测试通过 | [ROUND1_B](ROUND1_B.md) |
| C：W2 | 64 项 Template 测试、九完整目录 checker/dispatch 通过，19 YAML raw=preview | [ROUND2_C](ROUND2_C.md) |
| D：有限源复核 | 九更正 YAML、两 dispatch、九 Tolerance/十继承消费者及真实 from 通过 | [W2_REVIEW](W2_REVIEW.json) |
| D：共享 workflow | 187 个不同用例闭环；修复三条旧文档 pin 后 69 例回归通过，另一个 strict CLI 夹具显式选 strict 后单例通过；117 例不受影响结果复用 | [workflow 验证](round2_d/workflow_verification.json) |
| D：current/bridge | 九完整清单与索引一致，27 个实际目录 provenance 断言通过 | [终检](round2_d/current_verification.json) |

[Schema Wiki](../../schema_wiki/README.md) 保留原六项/15 YAML，并加入十二项/33 YAML，共 **18 项/48 YAML**。本批九 current 覆盖完整 19 YAML/9 dispatch，明确 `runner_p6_receipt:false`；另外三目录的 14 YAML/3 dispatch 仍匹配原 receipt。12 原 receipt、原 DELIVERY_INDEX 和固定 Source 未改，新清单不重签旧 P6。

共享接口没有新增门禁。默认 design/optimize 仍 write_only，strict 冻结实际 Template/规范并拒旧规则漂移；Extract bridge 透传原 CLI；标准 implement 保持原 AKA 委托，iterate 保持按版本计轮、keep 停止、失败反馈与未结算/异常停止及不重放。唯一额外产品兼容变更是经根划归的 row_softmax 公共材料清单三条文档 size/SHA 更新，保留 driver 身份检查。

695 个旧原件原位保全；277 个允许保护路径只有授权变化；12 state、supervisor、原 receipt 和用户 AGENTS 摘要相符。42 archive 元信息排除后未访问目标。历史公开仅五个旧候选和精确必要新证据，未公开 690 个旧原件、账户或 raw/native/思维捕获。[HISTORY_PUBLICATION](HISTORY_PUBLICATION.json) 给逐项去向与 allowlist；[公开历史说明](PUBLIC_HISTORY.md)明确 clone 不含旧运行账/冻结 run、未公开历史、原 /tmp 环境/fixture，SHA 不能代替缺失内容。

没有真实模型、GPU、Extract、monitor 调用、AKA 改动或新的 P6。masked 的 4616 例是旧索引证据复用，未重算。结构与有限来源检查不证明全部 33 稿、全域数值/性能、编译器寄存器分配或真实模型返工成本；比较器缺口、GDN 同步风险、mixed Sparse 哈希未确认和性能例外仍如实保留。批准计划的保留项和明确排除项均已收口，无额外执行轮或隐藏必做待办。
''')
(BASE/'README.md').write_text('''# KernelSchema main 整合与提取修订

两轮 A/B、C/D 的授权实施及独立验证已完成，当前为 **implementation_complete_ready_for_main_delivery**。根代理按 D 已审精确清单完成 Git 提交、正常推送与远端 SHA 核验。原批准字节、旧审计计划与原 receipt 保持历史原义。

- [最终实施报告](FINAL_REPORT.md)：实际变化、测试结果及限制。
- [实施覆盖](IMPLEMENTATION_COVERAGE.json)：26 发现、24 候选/104 出现、20 recap、11 既有修正、16 删减与 FC01/02。
- [公开历史说明](PUBLIC_HISTORY.md)及[公开 allowlist](HISTORY_PUBLICATION.json)：695 原件逐项保全、五旧候选及必要新证据，clone 的缺失材料边界。
- [W2 冻结对象](W2_SUBJECT.json)、[独立有限复核](W2_REVIEW.json)、[本批九 current 索引](../extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json)。
- [第一轮屏障](W1_GATE.json)、[第一轮收口](ROUND1_GATE.json)、[A](ROUND1_A.md)、[B](ROUND1_B.md)、[C](ROUND2_C.md)、[D](ROUND2_D.md)。
- [仲裁计划](ARBITRATION_PLAN.md)与[机器执行映射](execution_map.json)保留原计划快照；实际状态看[progress](progress.json)，唯一表外精确归属见[OWNERSHIP_CHANGES](OWNERSHIP_CHANGES.json)。

原 57 文件基线为 `41b7db0`，workflow 正常合并为 `6c3859e`，保留 37 个自然提交；W1/W3/W4 为 `1a221e4`、`d444101`、`f46b14a`。仅根负责 Git 写动作；AGENTS 用户改动、旧运行态与未公开捕获保留本地。没有真实模型/GPU/Extract/monitor 或 AKA 修改。
''')
p=BASE/'progress.json';d=json.loads(p.read_text());d.update(phase='implementation_complete_ready_for_main_delivery',execution_agents_started=4,round2_complete=True,w2_review_pass=True,current_verified=True,workflow_verified=True,mandatory_implementation_pending=[],next_action='root: verify exact reviewed stage allowlist, commit, normal fetch/push main, verify remote SHA');d['execution_agents'].update(C='m06_schema_corrections',D='m07_independent_integration');d['w2_subject']=ref(P+'W2_SUBJECT.json');d['w2_review']=ref(P+'W2_REVIEW.json');p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
subprocess.run(['/opt/venv/python3.13.12/bin/python','-B',str(OUT/'build_publication.py')],check=True,cwd=ROOT)
print('D reports and publication allowlist ready')
