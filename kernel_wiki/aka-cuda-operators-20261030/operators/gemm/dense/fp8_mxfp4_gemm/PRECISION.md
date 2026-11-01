# fp8_mxfp4_gemm 精度记录

固定域20/20、声明域20/20通过；数值判定沿用冻结checker，未放宽容差。每个输出和mutated input分别检查，同一alias可能对应两条记录，不能累加成独立错误元素数。

共40条数值检查，0条存在非零超差元素（均在该检查原有容许范围内）。最大单检查超差比例：0/8192 = 0.00000000%；位置：fixed/0，seed 2026101221，input.d。

| 域/shape | seed | 输出/输入 | 超差/总数 | 允许超差数 | relative L2 |
|---|---:|---|---:|---:|---:|
| 全部已测检查 | — | — | 0 | 按各检查合同 | 见ACCEPTANCE |

完整数值与容差见 [performance/ACCEPTANCE.json](performance/ACCEPTANCE.json) 的 fixed/domain.details。采样结果不代表全域零误差或 bitwise 相同。
