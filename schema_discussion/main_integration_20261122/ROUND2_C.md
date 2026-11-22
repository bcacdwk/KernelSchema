# 第二轮 C：W2 正文、规范与 current

2026-11-22。正文与规则已冻结，D 对同一实际字节的独立有限源复核通过，九份 current 与总索引已在该通过报告冻结后生成。冻结对象为 [W2_SUBJECT.json](W2_SUBJECT.json)，SHA256 `4936e08c5611c703497c6e53a244ff4e3cf746f993ebff62dce57c60752a2242`。

本次按批准的 W2 范围修改九 YAML，并分别将 Version 增一；Source 保留 `484902855fe7be82a74fe7c394503d1d7d222dbb` 固定来源。masked 补齐非空 tile 的 clamp 和 align16 公式，将 y 等式限定到有效行，分开 FP32 解码求值与实际 block-scaled 累加，保留额外行计算、窄化及写回。FMHA 只修四 lane 的 m/l 副本和 root 四 warp 的 page metadata 容量，decode page_regs 仍为 1024 B 活跃 payload。KDA 删除未读 state-index 指针必须 null 的断言，仍禁止启用间接寻址和 FP32 state，CHAIN 按 request 取 slot；两条窄化及原 Gap 未变。

五个 root 的九条 Tolerance 从五份固定 ACCEPTANCE JSON 恢复完整六字段，包括 mode、relative_to、显式 null 与完整 max 标签。所有记录保持 source-only、比较器定义缺口和不补默认值边界。只将 Tolerance 预算从 100/1 调整到 300 Unicode 字符/2 逻辑行；其他预算逐字保持原样。

DOC1 增加来源字段保全和两条异名登记短例；DOC4/5 明确逻辑 worker、物理先后接管、非同时活跃范围、旧存储与在途访问退休、独立副本容量及左到右短路。P2/P4 将未来基线线索归于“范围”，正式变体只直链真实完整基线。两份 dispatch 改为当前来源支持范围表述，直接保留 GDN 每线程 fence→tid0 arrival 缺 CTA 聚合的同步风险及对齐/vec==0 条件；mixed Sparse 明确最新原题回执缺失、最终题目哈希未确认，并保留比较器、逐输出、机制 A/B 缺口、SM103 采样、NCU 6/10 和四个性能例外。D 的两条定点反馈与证据补强已在本轮闭环，[修订记录](round2_c/independent_review_correction.json)。

实际 CPU 验证：

- 64 项 Template 定向测试通过，使用 `/opt/venv/python3.13.12/bin/python -B -m unittest tools.tests.test_yaml tools.tests.test_schema tools.tests.test_dispatch tools.tests.test_design_resolution`，Template 的 canonical `tools` 包在独立进程加载。新增用例覆盖完整记录、Unicode 300 字符/2 逻辑行边界、301 字符/3 行拒绝、输出绑定和其他预算不变。[命令与结果](round2_c/template_tests.json)、[输出](round2_c/template_tests.log)。
- 临时候选副本的九完整目录规范化均无差异。[候选检查](round2_c/candidate_normalization.json)。
- 九次原 `check_yaml.py <dir>/root.yaml --read-only --skip-hardware` 全部 exit 0；另外调用原 dispatch API，并复核十九份实际 YAML 字节等于预览，记录真实隐式 root 与显式 from 闭包。[检查、环境和日志索引](round2_c/structure_checks.json)。
- 五个固定 JSON 的 SHA/pointer/值核对一致；另外 24 YAML 与 12 receipt 摘要未变。[有限对照与保全](round2_c/tolerance_and_preservation.json)。masked 的 4616 个输入索引算术复用原有限证据，公式与来源未变，没有重新声称数值或 GPU 验证。

本次未运行模型、GPU、真实 Extract 或 monitor，未修改源码 kernel、旧 receipt、旧账/规则/运行态、AGENTS 或其他 worktree，未执行 Git 写动作。硬件字段显式跳过；编译器总寄存器分配、比较器缺口、来源同步风险、全域正确性与性能均不由本包证明。九份 current 均明确 `runner_p6_receipt:false`，不重签旧 P6。

九 current 覆盖完整十九 YAML/九 dispatch，保存 Version、固定 Source/源 SHA、真实隐式与显式依赖、规则/checker、旧 receipt。它们引用 [W2_REVIEW.json](W2_REVIEW.json) 的固定 SHA `af96e504979f3a98cd975f9f82922ad088821ad4a0c13ed0e217ea8d45679255`；该报告只引用先行 subject/source/checker，不反向引用未来 current。[CURRENT_SCHEMA_INDEX](../extract_schema_review_20261121/CURRENT_SCHEMA_INDEX.json) 只是九 current 的投影。生成顺序、命令与九份摘要见 [current_generation.json](round2_c/current_generation.json)，C 的只读一致性复核见 [current_consistency.json](round2_c/current_consistency.json)。D 对最终 current/index 的独立终检另记于其交付证据。
