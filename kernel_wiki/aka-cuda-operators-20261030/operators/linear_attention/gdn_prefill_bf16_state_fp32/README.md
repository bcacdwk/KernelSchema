# gdn_prefill_bf16_state_fp32

gdn prefill bf16 state fp32。

| 项目 | 说明 |
|---|---|
| 来源 | **参考实现** |
| 入口 | [candidate/solution.py](candidate/solution.py):`Model.forward`；参数及别名见[接口清单](../../../MANIFEST.json)。 |
| 初始化 | `Model(**init_kwargs)`；整个 [candidate/](candidate/) 一起复用。 |
| 题目 | [合同](problem/metadata.json) · [输入生成](problem/input.py) · [正确性参考](problem/reference.py)；`problem/solution.py` 是生产对照。 |
| 范围 | [固定用例](problem/shape_cases.json) · [支持范围](problem/shape_range.json)；容差、输出和状态约束以合同为准。 |
| 依赖 | filelock、torch、tvm_ffi；CUDA/C++ 工具链，版本与 flags 见源码构建入口。 |
| 性能 | [SUMMARY.json](performance/SUMMARY.json)，NCU kernel-sum（µs）；[原始测量](performance/ACCEPTANCE.json)。 |
| 状态 | [采样验收通过](performance/ACCEPTANCE.json)；[problem 待最终回执核验](../../../PROBLEM_NOTES.md)。 |
| 补充 | [PRECISION.md](PRECISION.md) · [来源与许可](LICENSES/)。 |

[统一规范](../../../CONVENTIONS.md)
