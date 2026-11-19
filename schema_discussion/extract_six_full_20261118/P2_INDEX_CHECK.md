# Sparse P2 dispatch 链接退修核对

结论：**本例是草案混用正式记录格式并遗漏当前基线链接，正常退修足够；不修改 Template 或阶段适配器。** 不能仅凭诊断中的 `current complete YAML` 推断工具要求 P2 完成 Design。本次检查实际检查文件登记和条目直链，没有验收未来 YAML 或 root 的完整 Design。

## 核对对象

- run：`8f0844f9-f279-4781-a19d-0216b7fc4ef5`（`dual_cache_latent_bf16`）。以下运行相对路径均位于 `.schema_extract/v2/runs/<run>/`。
- 失败 checkpoint：`dfcf3521a446548fd6b4c0bd95b5f5bcdc90d0ffa1bcaedd8f870eb4bf57896f`；其 `checkpoint.json` 记录 worker attempt `a0286ad4-c6e7-430f-be43-02c9d96aef29`，阶段 P2。
- 失败记录：该 attempt 的 `validated.json.check`，`kind=stage_snapshot`、`stage=P2`、`exit_code=1`。十项诊断均为条目缺少当前 YAML 基线直链；此 attempt 没有独立的 `check.json`。
- 固定规范：snapshot `e5db05dc281797bc4a7350cc1d7eb35b2d1331efb10f304bd0ab3b1955ff6388/rules/`。已逐字节确认其 P2、P4 prompt、prompt/common、DOC0、Template dispatch checker 和 Extract artifacts adapter 与当前源码相同。

## 实际稿件与退修原因

失败稿 `artifacts/dispatch.md:38` 已有 `[root](root.yaml)`；39–41 行将 `split_kv.yaml`、`tcgen05.yaml`、`tcgen05_packed.yaml` 登记为普通文本“仅计划”。目录中实际只有 `root.yaml`，因此完整设计登记本身满足早期阶段规则。

47 行却使用正式 `## 变体与组合`，其下十条全部采用 `### 名称（参数|局部|组合）`。49 行虽声明是 P2 轻量线索，每条正文仍没有任何当前 YAML 直链：

| 失败行 | 条目 | 稿件所指基线 |
| --- | --- | --- |
| 51 | Tile 几何与 plan 选择 | root 与未来 split_kv；root 仅为普通文本 |
| 57 | EI 早发重排 | root/split_kv，仅为普通文本 |
| 63 | DQ 输出 D 分区 | root，仅为普通文本 |
| 69 | Combine 形态与 partial 格式 | 仅未来 split_kv |
| 75 | tcgen05 tile 几何与次序 | 仅未来 tcgen05/tcgen05_packed |
| 81 | tcgen05 loader 与发射机制替代 | 仅未来 tcgen05/tcgen05_packed |
| 87 | Packing probe 门控与双流隐藏编排 | root 家族与未来 tcgen05_packed；无直链 |
| 93 | CUDA graph replay | root 与其他未来代表；无直链 |
| 99 | 注册 fastcall | 只指向 graph replay 的 host 路径，无 YAML 直链 |
| 105 | tcgen05 × split-KV | 仅未来 tcgen05/split_kv |

[P2 第 6 项](../../schema_extract/prompt/phases/02_dispatch_draft.md:20)允许暂拟对象和未来基线，但只链接实际文件，未来文件保留为计划。[DOC0 的正式格式](../../schema_docs/0.common.md:152)要求条目列明基线直链；[P4](../../schema_extract/prompt/phases/04_dispatch_plan.md:18)继续区分已有基线的正式记录与依赖未来正文的记录计划。纯未来基线线索可以保留在普通计划说明中，无需借用正式条目标题，也不应随意给它们补上不相关的 root 链接。

## 检查边界

[`validate_index`](../../schema_extract/tools/artifacts.py:430)对非最终阶段接受已链接的当前文件与“仅计划”的未来代表；只有 `final=True` 才要求全部所选文件存在并清除计划。随后 Template 的 [`dispatch.validate`](../../schema_template/tools/rules/dispatch.py:155)检查正式变体条目是否直接链接所传入的当前文件；它不读取 YAML，也不判断基线对象是否已经定义。

[`check_stage_snapshot`](../../schema_extract/tools/artifacts.py:677)实际传入 `stage=P2`，目录检查的 `final=False`；root 走 `_early_check`，不检查未来 Design。P4 对当前 root 执行完整定义检查，P6 检查整个实际集合及最终目录。本例没有阶段参数误传，也没有强迫 P2 创建三个未来文件。对象真实性、组合相容性及 P2 草案措辞仍须独立 reviewer 判断，链接通过不等于 Design 已完成。

## worker 已自行完成的退修

核对期间，原运行自行产生 worker attempt `31fc088a-8b08-4569-af51-3573ab9ca1de`，其 base 正是上述失败 checkpoint。新 checkpoint 为 `75e307ced21a9e2ca0248d06de8f81a8300c69fa4225784564af4ae894298cbb`：

- 四条纯未来基线内容移到 47–51 行“基线仅计划的记录线索”，保留机制、差异和依据。
- 六条涉及当前 root 的条目补上直接链接；其余未来基线继续用普通文本“仅计划”。
- `root.yaml` 与失败 checkpoint 字节完全相同，未为通过索引检查提前填写 Design。
- 该 attempt 的 `check.json` 已为 P2 `exit_code=0`、`diagnostics=[]`，同时明确 `yaml_files=[root.yaml]`，三个非 root 仍同时列于 `planned_nodes` 和 `missing_nodes`。

只读观察时运行已进入 P2 reviewer（attempt `7ed3ccba-aa5e-42a5-925a-f2c173495bca`）；这不是 P2 reviewer 已批准或后续阶段已完成的声明。格式退修由原 worker 完成，不需要 Supervisor 暂停、恢复或代改产物。

本核对仅新增本文。未改工具源码、schema 字段、冻结规范、运行状态或产物，未触发运行调用。这批冻结工具继续原规则；当前证据不支持增加阶段豁免。验证使用已有失败/通过检查记录、checkpoint 字节比较与有限代码阅读，没有重复运行检查或新增测试。
