package kernel

import "list"

// schema_docs/4: phase identity, work decomposition, placement and assignment.
#WorkSchedule: {
	thesis!: #Text
	flow!: {
		launch!: [#Launch, ...#Launch]
		handoff?: {
			[#Identifier]: #Handoff
		} & #NonemptyMap
	}
	phase!: {
		[#PhaseID]: #WorkPhase
	} & #NonemptyMap
	let Defined = [for id, _ in phase {id}]
	let Used = [for launch in flow.launch for id in launch.phase {id}]
	if !list.UniqueItems(Used) {
		flow: error("each phase must be submitted exactly once")
	}
	for id in Used if !list.Contains(Defined, id) {
		flow: error("launch references undeclared phase \(id)")
	}
	for id in Defined if !list.Contains(Used, id) {
		phase: error("phase \(id) is not submitted by flow.launch")
	}
	for field, handoffs in flow if field == "handoff" {
		for name, h in handoffs {
			if !list.Contains(Defined, h.producer) {
				flow: handoff: "\(name)": producer: error("unknown producer phase")
			}
			for consumer in h.consumer if !list.Contains(Defined, consumer) {
				flow: handoff: "\(name)": consumer: error("unknown consumer phase \(consumer)")
			}
		}
	}
}

#Launch: {
	phase!: [#PhaseID, ...#PhaseID] & list.UniqueItems
	order?: "per_worktile" | "per_worker" | "all" | "concurrent"
	start_after?: #ScheduleEvent | [#ScheduleEvent, ...#ScheduleEvent]
	protocol?: #Identifier
	if len(phase) > 1 {
		order!: "per_worktile" | "per_worker" | "all" | "concurrent"
	}
}

#ScheduleEvent: #Text & =~"^[a-z_]+\\.[a-z_]+(\\([^\\r\\n]+\\))?$"
#Handoff: {
	producer!: #PhaseID
	consumer!: [#PhaseID, ...#PhaseID] & list.UniqueItems
	storage!:    "global" | (string & =~"^tile_pipeline\\.phase\\.[a-z_]+\\.resource\\.[A-Za-z_][A-Za-z0-9_]*$")
	initialize?: number | #UpstreamReference
	ready?: #ScheduleEvent | [#ScheduleEvent, ...#ScheduleEvent]
	wait_at?: #ScheduleEvent | [#ScheduleEvent, ...#ScheduleEvent]
}

#WorkPhase: {
	describe!:      #Text
	decomposition!: #Decomposition
	mapping!:       #Mapping
	schedule!:      #Schedule
	if len(decomposition.partition) > 0 {
		schedule: worktile_order!: [#WorktileOrder, ...#WorktileOrder]
	}
	for field, order in schedule if field == "worktile_order" {
		let Dimensions = [for name, _ in decomposition.partition {name}]
		let Ordered = [for entry in order {
			if (entry.by & [...string]) == _|_ {entry.by}
			if (entry.by & string) == _|_ {for name in entry.by {name}}
		}]
		for name in Dimensions if !list.Contains(Ordered, name) {
			schedule: worktile_order: error("partition dimension \(name) is missing from worktile_order")
		}
		for name in Ordered if !list.Contains(Dimensions, name) {
			schedule: worktile_order: error("worktile_order names undeclared partition dimension \(name)")
		}
	}
}

#Decomposition: {
	strategy?: #Text
	partition!: {
		[#Identifier]: {
			size!: #PositiveExtent
			tile!: #PositiveExtent
			range?: [#PositiveExtent, #PositiveExtent | null]
		}
	} // {} is one whole-task worktile, not a missing-value placeholder.
	worktile?: {
		invariant?: {
			[#Identifier]: #InvariantBlock
		} & #NonemptyMap
		iteration?: {
			[#Identifier]: #IterationBlock
		} & #NonemptyMap
	} & #NonemptyMap
	target!: {
		single_tile!: {
			[#Identifier]: #DataBlock
		} & #NonemptyMap
		tile_reduction?: {
			dimension!: #Identifier
			input!:     #ReferenceOrList
			output!:    #ReferenceOrList
			method!:    #UpstreamReference
		}
	}
}

#DataBlock: {
	shape!: #Shape
	dtype!: #DType | #ResolvedDTypeRef
}
#InvariantBlock: B={
	#DataBlock
	offset?: [#Integer, ...#Integer]
	for field, offsets in B if field == "offset" {
		if len(offsets) != len(B.shape) {offset: error("offset length must match shape rank")}
	}
}
#IterationBlock: {
	#DataBlock
	direction!: #Direction
}

#UnitCount: {
	unit!:  #Text
	count!: #PositiveExtent
}
#Mapping: M={
	strategy?:  #Text
	worker!:    #UnitCount
	placement?: #UnitCount
	count!:     #PositiveExtent
	member?: [{[#Identifier]: #MemberRegion} & #NonemptyMap, ...{[#Identifier]: #MemberRegion} & #NonemptyMap]
	for field, members in M if field == "member" {
		if (worker.count & string) == _|_ {
			if len(members) != worker.count {member: error("member entries must match worker.count")}
		}
	}
}

#Schedule: {
	strategy?: #Text
	worktile_order?: [#WorktileOrder, ...#WorktileOrder]
	worker_assignment!: {
		preassigned?: #Preassigned
		refill!: "none" | ({[#Identifier]: #Text} & #NonemptyMap)
		exit!: #ReferenceOrList
	}
}
#WorktileOrder: {
	by!: #Identifier | ([#Identifier, ...#Identifier] & list.UniqueItems)
	group?:     #PositiveExtent
	direction!: #Direction
}
#Preassigned: {
	start!: #Extent
	step!:  #PositiveExtent
} | {
	start!: #Extent
	count!: #Extent
}
