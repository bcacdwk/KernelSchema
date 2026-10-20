# Four AKA problems and selected solutions

Export date: 2026-10-19. Target GPU: NVIDIA L20D / SM103.

| Directory | Scored shapes | Selected AKA | Production / AKA NCU kernel sum | Faster shapes |
| --- | ---: | --- | ---: | ---: |
| fa4_prefill | 30 | v37, original package label v43 | 1.1990x | 30/30 |
| fa4_decode | 30 | v55 | 1.1126x | 20/30 |
| qsa_prefill | 4 | **BF16 FIT v3, fc30f74** | **4.2724x** | **4/4** |
| gdn_prefill | 30 | Expert BF16/M64, tail-padding fix | 1.1591x | 25/30 |

Each problem directory contains the original reference.py, input.py,
metadata.json, shape_train.json, shape_valid.json, original solution/,
and the complete selected aka_solution/ source tree. shapes.json is the
scored shape_valid.json (an identical alias is added where needed).
Original root solution.py files are preserved as the task's baseline entry.
The original solution/ directories also include extracted reference/seed
kernel sources (FA4 Prefill FA4 seed and FA4 Decode CuTeDSL seed); those
must not be confused with the TRT-LLM Production baseline used for NCU.
For FA4 Prefill/Decode, production_ncu.py is the explicit measured
FlashInfer TRT-LLM Production adapter with reused Plan A counter buffers.
For QSA and GDN, the NCU baseline is solution/solution.py. Version identities
and separate entry-point roles are in MANIFEST.json.

All four problems have both Production and AKA implementations. GDN's
independent original input generator also has its existing Agate-compatible
copy under gdn_prefill/solution/agate_reference/; its construction script
and documentation are retained. Add the chosen aka_solution/ directory
to PYTHONPATH when using an external loader, and include that entire
directory for multi-file submissions. Dependencies are listed in each
solution/requirements.txt and aka_solution/requirements.txt; the FA4
production_ncu.py adapters use requirements-production-ncu.txt.

QSA is the BF16 FIT v3 tested on 2026-10-19, not the original v7. Its device
code and tested wrapper are unchanged. It passed 4/4 shapes and 16/16 cases.
The frozen implementation is validated for eager calls; CUDA Graph replay
needs ticket/tbase lifetime adaptation. See qsa_prefill/aka_solution/README.md.

GDN's fix narrows static_tma_full_chunks from T % 64 == 0 to T % 128 == 0,
retaining neutral padding (gate=1, beta=0). This prevents final_state from
being zeroed on an odd number of complete 64-token chunks, e.g. T=14784.
It retains M64 for all shapes and passed 30/30 shapes, 180/180 cases.

Performance values above are geometric means over the complete scored
shape sets, from each implementation's paired Production comparison.
QSA was measured on 2026-10-19; FA4 on 2026-10-14; GDN on 2026-09-22.
Jobs and clock settings differ between rows. Kernel sum includes required
preprocessing/combine kernels, excludes host overhead, and does not imply
every shape or complete API call is faster. Original validation records
are retained in the solution trees; NCU evidence is in evidence/.

MANIFEST.json lists every file's SHA-256, size and workspace provenance.
Run `python3 verify_package.py` after extraction to validate the archive.
Caches, .git, .DS_Store and Python bytecode are excluded.
