# Existing catalog relevance

The development baseline `484902855fe7be82a74fe7c394503d1d7d222dbb` already
contains the earlier six-operator Schema work. This material task does not
inspect, alter or re-extract the current new extraction tasks.

The stable Template examples include attention (FA4 prefill/decode, QSA
prefill) and GDN prefill. They illustrate scheduling, online normalization and
state/lifetime notation, but their fused attention/recurrent math, layouts and
CuTe/device-specific mechanisms differ from this four-shape Triton row-softmax
Task. This example does not assert that the catalog already contains a qualified
softmax neighbor. Design input includes this relevance note and DOC0–6, not a
whole catalog/source tree or private extraction sessions.

The directly relevant external reference is the pinned official Triton fused
softmax tutorial. Reading that public implementation is allowed in the stated
demo information boundary. Pure Schema reconstruction would need its own
explicit allowlist and should not claim that reference-assisted work satisfied
that different information boundary.
