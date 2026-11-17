# AKA solution: Decode v55

这是 02 Decode Attention 战役最终接受的 v55，原生匹配本题的 FP8 E4M3、HND、page64 和 13 参数 ABI，不包含 BF16/page128 转换桥。

- 入口：`solution.py::Model`
- 候选 commit：`b38f3c32e2abf33117c229412b32b27189df73a7`
- 晋升 commit：`815eead1eb7b65f29bb8a655ff5c9dc108b59df3`
- `gqa_decode_paged.py` SHA-256：`ec930c32758ea4f34e2df9fefa2de90f3ef345dc11fa66948fef74ae016fbe92`
- Agate L20D 复验 job：`dv_76824fda934c`
- 正确性：30/30 shapes passed，`atol=0.06`、`rtol=0.04`
- 复验最大 absolute/relative diff：`0.03125 / 0.09765625`

必须让 `qoder_greenfield_decode_attention/` 与 `solution.py` 同目录；它是该解的完整源码闭包。`CAMPAIGN_V55.json` 是原始接受记录，`AGATE_VALIDATION.json` 是本次独立复验的逐 shape 结果。

