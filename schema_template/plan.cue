package kernel

import "list"

// Public authoring entry: plans: kernel.#Task. Each key is one output file ID.
// No wrapper, helper, resolved Contract copy or validation metadata is exported
// into the individual plan YAMLs.
#Task: Task={
	root!: #RootPlan
	[ID=(#NodeID & !="root")]: #DerivedPlan & {
		meta_data: {
			name:     Task.root.meta_data.name
			describe: Task.root.meta_data.describe
			dispatch: id: ID
		}
	}
	if len(Task) > 1 {
		root: meta_data: dispatch!: #Dispatch
	}
	let Units = [for name, _ in root.contract.hardware_specification.execution {name}]
	for id, node in Task {
		for phase, p in node.design.work_schedule.phase {
			if !list.Contains(Units, p.mapping.worker.unit) {
				"\(id)": design: work_schedule: phase: "\(phase)": mapping: worker: unit: error("worker unit is not registered in hardware execution")
			}
			for field, placement in p.mapping if field == "placement" {
				if !list.Contains(Units, placement.unit) {
					"\(id)": design: work_schedule: phase: "\(phase)": mapping: placement: unit: error("placement unit is not registered in hardware execution")
				}
			}
		}
	}
}

#RootPlan: {
	meta_data!: #MetaData & {
		dispatch?: {
			id?:     "root"
			parent?: _|_
		}
	}
	contract!: {
		algorithm_requirement!:  #AlgorithmRequirement
		hardware_specification!: #HardwareSpecification
	}
	design!:      #Design
	realization?: #Realization
}
#DerivedPlan: {
	meta_data!: #MetaData & {
		dispatch!: {id!: #NodeID & !="root"}
	}
	design!:      #Design
	realization?: #Realization
}

#Design: {
	math_reshape!:  #MathReshape
	work_schedule!: #WorkSchedule
	tile_pipeline!: #TilePipeline
	let Work = [for id, _ in work_schedule.phase {id}]
	let Pipe = [for id, _ in tile_pipeline.phase {id}]
	for id in Work if !list.Contains(Pipe, id) {tile_pipeline: phase: error("missing tile_pipeline phase \(id)")}
	for id in Pipe if !list.Contains(Work, id) {work_schedule: phase: error("missing work_schedule phase \(id)")}
}
