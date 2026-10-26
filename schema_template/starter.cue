package starter

import kernel "kernelschema.local/schema_template:kernel"

// Fill the typed holes using schema_docs/1–6. A typed hole prevents export.
// init copies this file as package plan. The separate starter package keeps
// these holes out of the kernel definitions and descendant plan packages.
// Rename Input, Output and main to the actual objects and phase.
// Omit optional sections until there is real information to put in them.
plans: kernel.#Task & {
	root: {
		meta_data: {
			name:     string
			describe: string
			version:  1
		}
		contract: {
			algorithm_requirement: {
				input: Input: {meaning: string, shape: [{axis: int | string}], dtype: string}
				output: Output: {meaning: string, shape: [{axis: int | string}], dtype: string}
				compute: [{name: string, math: string}]
				golden_reference: {source: string, tolerance: Output: string}
			}
			hardware_specification: {
				device: string
				arch:   string
				execution: Unit:    string
				feature: Mechanism: string
				stack: {language: string, toolchain: string}
				wiki: {
					hardware: string
					programming: language: string
					microbench: string
					sota: {kernel: string, schema: string}
					profile: string
				}
			}
		}
		design: {
			math_reshape: {
				thesis: string
				rewrite: [{step: "compute", name: string, math: string}]
			}
			work_schedule: {
				thesis: string
				flow: launch: [{phase: ["main"]}]
				phase: main: {
					describe: string
					decomposition: {
						partition: Dimension: {size: int | string, tile: int | string}
						target: single_tile: Output: {shape: [{axis: int | string}], dtype: string}
					}
					mapping: {worker: {unit: string, count: int}, count: int | string}
					schedule: {
						worktile_order: [{by: "Dimension", direction: string}]
						worker_assignment: {refill: string, exit: string}
					}
				}
			}
			tile_pipeline: {
				thesis: string
				phase: main: lane: Main: {
					info: member: string
					pipe: steady: [string]
				}
			}
		}
	}
}
