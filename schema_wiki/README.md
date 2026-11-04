# Schema Wiki

本目录按 [kernel_wiki/catalog.json](../kernel_wiki/catalog.json) 的 44 条 `paths.readme` 父路径组织初始输出。六个算子已通过终审并发布，当前推荐版本合计 77 个节点 YAML；[修订版本](revisions/README.md)与原始快照分别保留，未提取的叶目录继续使用空 `.gitkeep` 占位。来源、验证范围和审计记录见[六类算子试用记录](../schema_extract/TRIAL_REPORT.md)。

| 分类 | 输出目录 | 算子数 |
| --- | --- | --- |
| GEMM | [gemm](aka-cuda-operators-20261030/operators/gemm/)（dense / grouped / grouped_masked） | 15 |
| MoE | [moe](aka-cuda-operators-20261030/operators/moe/) | 5 |
| FMHA | [fmha](aka-cuda-operators-20261030/operators/fmha/) | 8 |
| MLA | [mla](aka-cuda-operators-20261030/operators/mla/) | 4 |
| 稀疏注意力 | [sparse_attention](aka-cuda-operators-20261030/operators/sparse_attention/) | 4 |
| 线性注意力 | [linear_attention](aka-cuda-operators-20261030/operators/linear_attention/) | 8 |

已交付结果（当前推荐版本）：

| 算子 | 节点数（含 root） |
| --- | ---: |
| [fp8_scaled_grouped_gemm](revisions/2.2.4/aka-cuda-operators-20261030/operators/gemm/grouped/fp8_scaled_grouped_gemm/dispatch.md) | 3 |
| [nvfp4_fused_moe](aka-cuda-operators-20261030/operators/moe/nvfp4_fused_moe/dispatch.md) | 60 |
| [fmha_mxfp8_qkv](aka-cuda-operators-20261030/operators/fmha/fmha_mxfp8_qkv/dispatch.md) | 8 |
| [mla_expanded_fp8](aka-cuda-operators-20261030/operators/mla/mla_expanded_fp8/dispatch.md) | 1 |
| [dual_cache_latent_bf16](aka-cuda-operators-20261030/operators/sparse_attention/dual_cache_latent_bf16/dispatch.md) | 3 |
| [gdn_prefill_bf16_state_bf16](aka-cuda-operators-20261030/operators/linear_attention/gdn_prefill_bf16_state_bf16/dispatch.md) | 2 |

算子说明、源码与证据从 [Kernel Wiki 总索引](../kernel_wiki/CATALOG.md)进入。输出目录只保存 `root.yaml`、计划内的派生 YAML、`dispatch.md` 和发布回执；提取方法见[流程总览](../EXTRACT_SCHEMA.md)与 [Supervisor 手册](../schema_extract/README.md)。

从仓库根目录按 catalog 查得输出路径，再执行一个算子的提取：

```sh
operator_id=bf16_gemm
schema_output=$(python3 - "$operator_id" <<'PY'
import json
from pathlib import Path
import sys

catalog = json.loads(Path("kernel_wiki/catalog.json").read_text())
operator = next(item for item in catalog["operators"] if item["id"] == sys.argv[1])
print(Path("schema_wiki") / Path(operator["paths"]["readme"]).parent)
PY
)
python3 schema_extract/extract.py extract --id "$operator_id" --output "$schema_output"
python3 schema_template/check_yaml.py --skip-hardware "$schema_output/root.yaml"
```

Runner 仅消费叶目录内唯一且为空的普通 `.gitkeep`，随后原子发布已审成果。已有内容会保留并报告冲突，同一完整成果可幂等重放。提取保留硬件模板占位，检查必须显式使用 `--skip-hardware`。
