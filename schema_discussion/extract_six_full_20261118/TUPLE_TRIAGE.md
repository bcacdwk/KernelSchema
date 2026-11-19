2026-11-18：六算子 tuple 候选独立语义分流（Supervisor 辅助，只读核查）

结论：文档示例不足可以合并为一个非阻塞框架待办；MLA 当前获批 Contract 的两个 tuple 条目仍需由原 Qwen worker/reviewer 做 P1 限域维护。无需修改字段规则、增加封闭类型表或放宽 `python_none` 检查。本报告没有代写或批准 schema，也没有修改运行状态、checkpoint、回执或工具。

核查对象是 MLA run `0d78eac5-1e45-4ac0-9725-ab885b5c52dc` 的已批准 P1 checkpoint `4ce38e73b7cb8587d907df19f329434f135b1193a190ba6f7cc65dbf50765957`，其中 `root.yaml` SHA-256 为 `eb60bb4bae117a4c6b72d082d943c7099778827c1abf3e00098bc033ff44d9e0`。读取时该 run 处于 P2 worker。另核对 GEMM run `06f3ed2c-daa0-46f0-a0dd-a7e1f4f4df31` 的 P1 checkpoint `8175006fc5968a5e57c3bdd5197107716082e722dd4b587fa9b85c7d76de945c`。当前 common、DOC1、`symbols.py` 与固定提交 `24d2b3d3ca9d4907c4e7efd9607cf78cfb1a52b2` 内容相同。

1. **候选有成立部分，也有开放 dtype 规则误读。** [DOC1](../../schema_docs/1.contract.algorithm_requirement.md) 第 58 行明确 dtype “不是封闭枚举”，第 71 行允许在 `meaning` 或 `Math` 中解释已定义类型；[common](../../schema_docs/0.common.md) 第 223–225 行区分固定结构与开放取值，填写开放取值属于实例填写。因此，不能仅因 `python_tuple` 没有出现在例子或所谓登记表中便判定非法；使用实际已知的 Python tuple 类型、明确其值语义不需要新增字段。不过，随意起一个未定义的名字仍不够，容器类型与元素类型必须分清。候选提出的 `python_tuple + shape[element:4]` 也不能自动当成规范答案，因为该写法还需说明 dtype 指向整容器还是每个元素。缺少固定非 Tensor 序列、框架 dtype 值的示例这一事实成立。

2. **MLA 第 33 行存在实际矛盾，不能靠说明字段补救。** [当前获批 root](../../.schema_extract/v2/runs/0d78eac5-1e45-4ac0-9725-ab885b5c52dc/checkpoints/4ce38e73b7cb8587d907df19f329434f135b1193a190ba6f7cc65dbf50765957/artifacts/root.yaml) 的 `sage_attn_sfs` 写 `shape: []`、`dtype: python_none`，同时在 `meaning/default` 写四元 tuple。DOC1 第 210 行明定 `python_none` 是“仅取 Python None 的单值类型”；它不能表示 `(None,None,None,None)`。这不是单凭 `shape: []` 判错：完整的非数值对象可以由明确类型作为一个整体描述，真正冲突是借用已有 singleton 类型却填入另一类对象。`meaning/default` 保留了人工可读的 tuple 事实，但没有取消 dtype 的正式含义，故不能称该条在现有规则下完整成立。

3. **相邻整数 tuple 也要核对固定值域。** 第 34 行 `num_elts_per_sage_attn_blk` 的 `[element:4] + python_int` 可理解为有四个逻辑整数元素的序列，不能仅因它与非数值 tuple 写法不同就判为不一致。但它只有 `default: '(0,0,0,0)'`，没有 `range` 或 `Compute` 的固定值约束；DOC1 第 200–208 行规定默认值不限制显式传入值，整数默认值域为全部可表示值，不能从 `meaning` 推定更窄范围。固定 `problem/input.py:106–129` 的构造链没有给两个 tuple 赋其他值，`_make_inputs:129` 始终传入全 None 四元 tuple 和全零四元 tuple；`candidate/solution.py:2105–2106` 与 `problem/reference.py:78` 也保留这两个 tuple 默认值。因此应在本次两个条目的限域维护中明确冻结接口对象及其实际取值，不能把默认值当成固定域的替代。GEMM 对应条目已有全对象等值 `range`，未发现同一问题。

4. **影响与最小修复边界。** 两个参数在本任务参考计算和 candidate 的后续调用中不参与 attention 数值结果，因此没有证据表明数学、输出数值或 Design 因此错误。但 Contract 是真实调用接口的登记，tuple 和单个 None 是不同对象；原样发布会保留类型矛盾，并把固定整数 tuple 的有效域表达得过宽。建议走现有 maintain 的 P1 `review`，授权范围只含下面两个 Input 条目；需要修订时由原 Qwen worker 用现有 `meaning/shape/dtype/range/default` 表达实际类型与固定取值，再由独立 Qwen reviewer verify。类型名及具体排列由该流程决定；可考察把已知非数值 tuple 对象作为一个有明确定义的整体值登记，不把 `python_none` 重定义为 tuple，不要求两个不同元素类型的 tuple 强行使用同一 shape 写法。P1 维护不替代后续正常阶段及 P6。

建议的公开维护任务内容（仅建议，未提交）：

```json
{
  "issue_id": "tuple-abi-input-typing",
  "tasks": [
    {
      "stage": "P1",
      "node": "root",
      "action": "review",
      "fields": [
        {
          "path": "root.yaml",
          "fields": [
            "CONTRACT.Algorithm_requirement.Input.sage_attn_sfs",
            "CONTRACT.Algorithm_requirement.Input.num_elts_per_sage_attn_blk"
          ]
        }
      ],
      "reason": "核对固定 input.py:129、candidate/solution.py:2105–2106、reference.py:78 与 DOC1:58,71,200–210：当前 sage_attn_sfs 的 python_none 单值类型与四元 tuple 的 meaning/default 冲突；相邻整数 tuple 仅设 default，未表达固定全零有效值。按既有开放 dtype 及现有字段定义实际 tuple 对象、元素含义和冻结取值；default 只表示省略时取值，不据签名扩大冻结接口。无需改 schema_docs、checker、其他 ABI、Compute 或 Design；由原 worker/reviewer 修订并核验。"
    }
  ]
}
```

5. **两个框架 ID 的文档部分同根，应合并处理。** 保留已有 `non-tensor-parameter-registration` 作为“固定非 Tensor 序列和框架 dtype 值登记示例”的唯一规范待办，补入 MLA 例子，并明确开放 dtype 与 `python_none` 单值类型边界。`tuple-abi-input-typing` 原候选和确认回执保留稳定 ID；在本批分流决定中记录其文档部分并入前者、“没有受支持方式/未登记类型一律非法”论断不成立、当前 MLA 实例矛盾另由同 run 限域维护解决。无需为同根示例缺口再增加一个并行语言设计项目；也不能因合并文档 issue 而跳过 MLA 产物修正。两种处理的完成状态分别记录。

原回执证据定位（不转录模型推理或原生日志）：

| 算子 | worker attempt | reviewer attempt | 稳定候选 ID |
| --- | --- | --- | --- |
| GEMM | `cf14892a-8c64-47af-b944-78d741b45a8f` | `f799489b-6ce0-4c0a-afe2-8681f9c78200` | `non-tensor-parameter-registration` |
| MLA | `2768965d-56c6-40e5-834d-07e4488370f4` | `5cb325e9-a33f-41cd-b690-33c5e1ba456b` | `tuple-abi-input-typing` |

各回执位于 `.schema_extract/v2/runs/<上述 run>/attempts/<attempt>/receipt.json`；已核对 `framework_candidates` 及 MLA reviewer `decision: approve`。确认标签仅证明 reviewer 提报过该候选，不自动证明其全部解释成立。长期 issue 的可复查依据应继续使用固定 Git 版本的 DOC1/common、上述输入与签名路径，不仅依赖临时运行目录。

Supervisor 采纳范围：确认 sage_attn_sfs 的 tuple/None 类型矛盾。整数 tuple 的固定全零域不能仅由输入生成样本或签名默认值推出；已提交维护任务要求原 reviewer 核对真实任务材料，只有明确依据才收窄 range。该条目前是有界核对事项，不预设必须改成固定全零域。
