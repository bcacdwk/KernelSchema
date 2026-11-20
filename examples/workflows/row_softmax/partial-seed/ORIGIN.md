Manual implementation fixture authored by Codex during B04 on 2026-11-20,
for an explicit nonempty partial seed. This identifies its actual author;
it is not a claimed output of a separate workflow model run or a human engineer.
It contains ABI/shape scaffolding only and intentionally raises before writing
out. No model/GPU call, benchmark or qualification produced these bytes. Freeze
it with Publication.put_source and pass its exact Ref as seed. The Task baseline
remains the separately frozen evaluator/baseline.py.
