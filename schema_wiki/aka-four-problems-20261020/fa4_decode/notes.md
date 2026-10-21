# fa4_decode schema

本目录包含 7 份独立 schema，按 query 长度和 dispatch 类别划分。仅阈值、split 数或循环长度不同的配置在同一份 schema 内表达。

| Query 长度 | Dispatch 类别 | Schema |
| --- | --- | --- |
| 1 | direct | [q1_direct](fa4_decode__q1_direct.yaml) |
| 1 | cluster | [q1_cluster](fa4_decode__q1_cluster.yaml) |
| 1 | global | [q1_global](fa4_decode__q1_global.yaml) |
| 4 | direct | [q4_direct](fa4_decode__q4_direct.yaml) |
| 4 | cluster | [q4_cluster](fa4_decode__q4_cluster.yaml) |
| 4 | global | [q4_global](fa4_decode__q4_global.yaml) |
| 4 | compact | [q4_compact](fa4_decode__q4_compact.yaml) |

每份 schema 包含 `meta_data`、`contract`、`design`、`realization`。`dispatch.id` 为 `fa4_decode.<文件后缀>`；精确路由条件见 `meta_data.dispatch.when`，工作参数见 `design.work_schedule`。
