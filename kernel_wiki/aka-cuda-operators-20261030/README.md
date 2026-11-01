# CUDA 算子题库：44 题

六类算子，20 个 AKA 解、24 个参考实现。每题的 candidate、problem、performance 放在一起。

| 分类 | 题数 |
|---|---:|
| [矩阵乘法 GEMM](operators/gemm/README.md) | 15 |
| [融合专家计算 MoE](operators/moe/README.md) | 5 |
| [多头注意力 FMHA](operators/fmha/README.md) | 8 |
| [多头潜在注意力 MLA](operators/mla/README.md) | 4 |
| [稀疏注意力](operators/sparse_attention/README.md) | 4 |
| [线性注意力](operators/linear_attention/README.md) | 8 |

入口统一为 `candidate/solution.py:Model.forward`；具体参数、初始化钩子见 [MANIFEST.json](MANIFEST.json) 的 interface。problem/solution.py 是生产对照，problem/reference.py 是正确性参考。

每题先读 README，再按需查看 problem 合同和 performance/SUMMARY.json。摘要保留计时方法、对照身份、缺测和人工例外；原始性能证据保持原样。

[统一规范](CONVENTIONS.md) · [problem 对应证据](PROBLEM_AUDIT.json) · [三题待核验备注](PROBLEM_NOTES.md)

检查：`python3 tools/verify.py`；元数据转换测试：`python3 tools/test_metadata.py`。本次规范化未重跑 GPU 评测；通用评测执行器由运行环境提供。
