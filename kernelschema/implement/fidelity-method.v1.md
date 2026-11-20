# Current-Schema finite review v1

Read the complete fixed Schema, root Contract and selected scope. Only selected
design nodes require implementation; actual dispatch for every selected Task
partition is mandatory. The host enumerates current mathematical/precision,
ownership/schedule, memory/lifetime/pipeline, Require and Forbid obligations.
Every item must be assessed independently against the exact supplied kernel.py.
Unknown applicable requirements stay unknown. Never replace the current design
with the original example's one-row ownership constants. Numerical correctness
and speed do not prove fidelity. Omitting subtract-max may remain numerically
correct on [-20,20] and still violate a selected stable-normalization design.

Return kind=review content with exactly items and coverage. items is an array
with one object per supplied obligation: id, status (pass/fail/unknown), reason,
source_locations, evidence. A location is kernel.py:L<first>-L<last> (or a single
line); evidence is [{location, quote}], quoting those exact complete source lines
without a trailing newline. Each pass/fail requires nonempty exact evidence.
coverage is an array with one object per supplied mapping: coverage_id, node,
variant, source_locations, evidence, status, reason. Coverage tests implemented
input partition routing independently from design fidelity. Do not report a
design blocker from ordinary code/compile/numerical/fidelity defects or unknown.
No tools, source edits, evaluator private files, planner or permission changes.

Machine analysis only proves narrow source facts, including a directly called
forbidden torch.softmax or explicit low-precision conversion when disallowed.
It does not establish complete control flow, mathematical equivalence or memory
safety. The independent read-only reviewer must justify all passing obligations;
missing, invented, inconsistent or unlocated answers yield unknown. Reports bind
current Q/Task/R/rules/scope/C, this method's exact bytes, original reviewer
receipt, actual runtime/model/implementation/scope identity and all limitations.
