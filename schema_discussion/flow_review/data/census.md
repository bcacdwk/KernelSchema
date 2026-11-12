# Flow 机械盘点

统计固定输入中的字段使用，不等于当前规则的语义批准；候选稿与已批准稿保留各自状态。

## 十个 root

| 算子 | 状态 | phase | launch | handoff | 位置 |
| --- | --- | ---: | ---: | ---: | --- |
| template_gdn_prefill | template | 2 | 2 | 1 | [150](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_gdn_prefill/root.yaml:150) |
| template_fa4_prefill | template | 1 | 1 | 0 | [143](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_prefill/root.yaml:143) |
| template_fa4_decode | template | 2 | 2 | 3 | [206](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_fa4_decode/root.yaml:206) |
| template_qsa_prefill | template | 2 | 2 | 2 | [151](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/template_qsa_prefill/root.yaml:151) |
| extract_gdn | approved | 2 | 2 | 8 | [326](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gdn/root.yaml:326) |
| extract_gemm | approved | 1 | 1 | 0 | [200](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_gemm/root.yaml:200) |
| extract_mla | approved | 1 | 1 | 0 | [291](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_mla/root.yaml:291) |
| extract_sparse | approved | 2 | 2 | 3 | [283](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_sparse/root.yaml:283) |
| extract_fmha | approved | 1 | 1 | 0 | [390](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_fmha/root.yaml:390) |
| extract_moe | approved | 7 | 7 | 11 | [375](/home/shaohanyong.shy/KernelSchema/schema_extract/archive/20261109-ten-operator-text-audit/inputs/extract_moe/root.yaml:375) |

## 分支形态

同一算子按解析后 flow 的完整内容去重，ID 不同仍算不同，不能当作语义拓扑数量。

| 算子 | 有 flow 的文件 | 本地 flow | 不同 flow 内容 |
| --- | ---: | ---: | ---: |
| template_gdn_prefill | 2 | 2 | 2 |
| template_fa4_prefill | 11 | 11 | 2 |
| template_fa4_decode | 7 | 7 | 2 |
| template_qsa_prefill | 1 | 1 | 1 |
| extract_gdn | 6 | 6 | 6 |
| extract_gemm | 7 | 7 | 5 |
| extract_mla | 2 | 2 | 1 |
| extract_sparse | 8 | 8 | 4 |
| extract_fmha | 43 | 43 | 8 |
| extract_moe | 111 | 94 | 73 |

## 本地字段计数

只数 semantic_eligible 的本地 work_schedule；YAML alias 按字段实例计数，from_parent 不重复计入。

```json
{
  "files_parsed": 198,
  "by_category": {
    "template": 21,
    "approved": 145,
    "review_candidate": 13,
    "needs_authoring": 14,
    "attempted_without_authored_result": 5
  },
  "with_flow": 198,
  "local_flow": 181,
  "field_instances": {
    "launch.phase": 909,
    "launch.start_after": 726,
    "handoff.producer": 1325,
    "handoff.consumer": 1325,
    "handoff.storage": 1325,
    "launch.protocol": 637,
    "handoff.ready": 1310,
    "handoff.wait_at": 269,
    "launch.order": 3
  },
  "order": {
    "(single phase)": 906,
    "per_worker": 3
  },
  "protocol": {
    "pdl": 636,
    "shape_gated_pdl": 1
  },
  "storage": {
    "global": 1322,
    "on-chip reference": 3
  },
  "field_instances_by_category": {
    "template": {
      "launch.phase": 29,
      "launch.start_after": 8,
      "handoff.producer": 16,
      "handoff.consumer": 16,
      "handoff.storage": 16,
      "launch.protocol": 3,
      "handoff.ready": 9,
      "handoff.wait_at": 9
    },
    "approved": {
      "launch.phase": 729,
      "launch.protocol": 512,
      "launch.start_after": 594,
      "handoff.producer": 1093,
      "handoff.consumer": 1093,
      "handoff.storage": 1093,
      "handoff.ready": 1085,
      "handoff.wait_at": 159,
      "launch.order": 3
    },
    "review_candidate": {
      "launch.phase": 90,
      "launch.start_after": 77,
      "launch.protocol": 75,
      "handoff.producer": 132,
      "handoff.consumer": 132,
      "handoff.storage": 132,
      "handoff.ready": 132,
      "handoff.wait_at": 29
    },
    "needs_authoring": {
      "launch.phase": 61,
      "launch.start_after": 47,
      "launch.protocol": 47,
      "handoff.producer": 84,
      "handoff.consumer": 84,
      "handoff.storage": 84,
      "handoff.ready": 84,
      "handoff.wait_at": 72
    }
  }
}
```
