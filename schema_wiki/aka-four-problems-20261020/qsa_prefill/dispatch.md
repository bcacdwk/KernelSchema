# qsa_prefill

| Schema | Dispatch ID | 修订号 |
| --- | --- | --- |
| [qsa_prefill__union_persistent.yaml](qsa_prefill__union_persistent.yaml) | `qsa_prefill.union_persistent` | 2 |

适用条件：NVIDIA L20D / SM103，single-request fitted-step15，串行 eager 调用。具体输入、布局和有效域以 YAML 的 `meta_data.dispatch.when` 与 `contract` 为准。

`use_prefill_config` 和 `out` 是否传入均不改变 dispatch。

未确定项统一保留在 YAML 的 `@pending` 占位中。
