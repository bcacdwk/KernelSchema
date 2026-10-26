package plan

import kernel "kernelschema.local/schema_template:kernel"

// Synthetic validation fixture. The testing.invalid references deliberately
// identify test data; this is not a verified kernel design or hardware binding.
plans: kernel.#Task & {
	root: {
		meta_data: {
			name:     "scalar_copy"
			describe: "Copy one scalar; synthetic validation fixture."
			version:  1
		}
		contract: {
			algorithm_requirement: {
				input: X: {meaning: "Input scalar.", shape: [], dtype: "fp32"}
				output: Y: {meaning: "Output scalar.", shape: [], dtype: "fp32"}
				compute: [{name: "Copy", math: "Y = X."}]
				golden_reference: {
					source: "testing.invalid/kernel@fixture-v1:reference.py::copy"
					tolerance: Y: "exact"
				}
			}
			hardware_specification: {
				device: "Synthetic test device"
				arch:   "test"
				execution: {CTA: "One cooperative block.", warp: "One execution group."}
				feature: register: "Explicit per-lane local state."
				stack: {language: "Test language 1", toolchain: "Test compiler 1"}
				wiki: {
					hardware: "<testing.invalid/hardware@pending:test.md>"
					programming: language: "<testing.invalid/language@pending:test/>"
					microbench: "<testing.invalid/microbench@pending:test/>"
					sota: {
						kernel: "<testing.invalid/kernel@pending:test/>"
						schema: "<testing.invalid/schema@pending:test/>"
					}
					profile: "<testing.invalid/profile@pending:test/>"
				}
			}
		}
		design: {
			math_reshape: {
				thesis: "Keep the Contract computation and introduce no intermediates."
				rewrite: [{step: "copy", name: "Copy", math: "Y follows Contract compute Copy."}]
			}
			work_schedule: {
				thesis: "One worker produces the scalar result."
				flow: launch: [{phase: ["copy"]}]
				phase: copy: {
					describe: "Produce Y."
					decomposition: {
						partition: {}
						target: single_tile: Y: {shape: [], dtype: "fp32"}
					}
					mapping: {worker: {unit: "CTA", count: 1}, count: 1}
					schedule: worker_assignment: {
						preassigned: {start: 0, count: 1}
						refill: "none"
						exit:   "assigned_done"
					}
				}
			}
			tile_pipeline: {
				thesis: "One synchronous lane needs no cross-lane handoff."
				phase: copy: lane: Main: {
					info: member: "warp[0:1]"
					pipe: steady: ["produce:Y"]
				}
			}
		}
	}
}
