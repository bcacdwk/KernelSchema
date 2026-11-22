# Execution environment requirements

Materials target Linux x86_64, Python 3.11 source checkout/editable, a CUDA
NVIDIA GPU and the Triton/PyTorch stack. **No concrete device or runtime profile
has been selected or qualified here.** H20 is the first authorized candidate
to check, not an observed device or a portability guarantee. The materials CPU
checks on Python 3.13.12 do not qualify the planned Python 3.11 runtime.

Before freezing an executable Task/profile, record exact device name and
architecture, visible-memory/partition configuration when relevant, driver,
CUDA runtime/toolkit, Python, torch, Triton compiler/runtime, OS and execution
channel/adapter identity. Retain fixed evaluator/input/timing/source digests,
actual environmental report, timestamp and raw receipts. Record any image and
its digest when an image is used. Freeze model CLI/requested/reported model
identities separately for actual model work; they are not needed to read these
materials. A profile must not contain credentials.

Use official documentation matching the selected device and software versions.
The pinned Triton v3.5.1 tutorial is a reference snapshot, **not** a declaration
that v3.5.1 is the qualified runtime. Its applicability to the ultimately
selected runtime must be checked and frozen before executing a request.
Device-specific hardware, PTX/SASS, tooling/profiling and performance material
bindings are still missing; see references/missing.md. Planner and coder cannot
choose a different fixed device/stack silently. Changed Task identity requires
new applicable qualification.

Execution is inference-mode, eager, single-call-stream, with CUDA Graph and
concurrent streams unsupported. The candidate must preserve legal nonzero
offsets and operate on the supplied objects. Stage immutable candidate and
evaluator bytes, verify what was actually loaded and executed, and isolate the
candidate from evaluator writes. A compile target, document, historical result,
CPU check or available API is not evidence of current GPU/channel capability.

K1.evaluator will implement the trusted execution interface after K2.interface;
K1.real will qualify the chosen channel. This phase ran no GPU, model CLI or
Agate operation and has no native V0/ABI/timing/receipt qualification.
