package kernel

import (
	"list"
	"struct"
)

// Task and plan identity
// ----------------------

// Task validation entry: each key is the basename of one YAML plan file.
// The loader supplies this container; authors write individual YAML plans.
#Task: Task={
	root!: #RootPlan
	[ID=(#NodeID & !="root")]: #DerivedPlan & {
		// #MetaData/#Dispatch require these fields. Optional equalities check
		// supplied identities without filling missing YAML values.
		meta_data: {
			name?:     Task.root.meta_data.name
			describe?: Task.root.meta_data.describe
			dispatch?: id?: ID
		}
	}
	if len(Task) > 1 {
		root: meta_data: dispatch!: #Dispatch
	}
	let Units = [for name, _ in root.contract.hardware_specification.execution {name}]
	for id, node in Task {
		for pid, p in node.design.work_schedule.phase {
			if !list.Contains(Units, p.mapping.worker.unit) {
				"\(id)": design: work_schedule: phase: "\(pid)": mapping: worker: unit: error("worker unit is not registered in hardware execution")
			}
			for field, placement in p.mapping if field == "placement" {
				if !list.Contains(Units, placement.unit) {
					"\(id)": design: work_schedule: phase: "\(pid)": mapping: placement: unit: error("placement unit is not registered in hardware execution")
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

// schema_docs/1: task identity is separate from dispatch identity and revision.
#MetaData: {
	name!:     #Text
	describe!: #Text
	source?:   #Source
	version!:  #PositiveInt
	dispatch?: #Dispatch
}

#Dispatch: {
	id?:     #NodeID
	parent?: #NodeID
	when!:   #Text
}

// Shared values and notation
// --------------------------

// Shared notation. Text is deliberately not an executable mathematical language.
// See schema_docs/1 §语法规则. English prose and meaningful content are reviewed.
#Text:           string & =~"\\S" & !~"^\\s*-\\s*$"
#Identifier:     string & =~"^[A-Za-z_][A-Za-z0-9_]*$"
#NodeID:         string & =~"^[a-z_][a-z0-9_]*$"
#PhaseID:        string & =~"^[a-z_]+$"
#NonemptyMap:    struct.MinFields(1)
#PositiveInt:    int & >0
#NonnegativeInt: int & >=0

// Missing-evidence notation is reserved for wiki, not Contract device facts.
#HardwareFact: #Text & !~"<[^<>]+@[^<>]+:[^<>]+>" & !~"^unknown:"

// Only structured integer-expression fields use this notation. The companion
// validator checks its small arithmetic grammar; it never executes expressions.
#IntegerExpression: #Text & =~"^[A-Za-z0-9_+*/%(), \\t\\-]+$"
#Extent:            #NonnegativeInt | #IntegerExpression
#PositiveExtent:    #PositiveInt | #IntegerExpression
#Integer:           int | #IntegerExpression
#KnownValueText:    #Text & !~"^(unknown:|ref:|bind:)" & !~"^<[^<>]*>$"
#Scalar:            number | bool | #KnownValueText
#Literal: #Scalar | [#Literal, ...#Literal]

// Dtypes remain open vocabulary. References and missing-value markers have
// separate types, so an arbitrary string is not silently accepted as a dtype.
#DType: #Identifier & !="not_applicable"
#Shape: [...#Axis]
#Axis: {
	[#Identifier]: #Extent
} & struct.MinFields(1) & struct.MaxFields(1)

#Direction: "ascending" | "descending"
#Mode:      "sync" | "async" | "ordered"
#Range: [#Integer, #Integer]

// Iteration domains may be signed; member and storage coordinates may not.
_#Region: [#Extent, #Extent]
#MemberRegion: _#Region | [_#Region, ..._#Region]

// A reference's existence, immutable revision and relevance require evidence.
// A syntactically valid reference is not a claim that its target was fetched.
#RepositoryReference: #Text & =~"^[^@<>\\s]+@[^:<>\\s]+:[^<>\\r\\n]+$"
#BoundReference:      #RepositoryReference & !~"@pending:"
#ReferenceEntry:      #BoundReference & =~"::[A-Za-z_][A-Za-z0-9_.]*$"
#WikiReference:       #BoundReference | (#Text & =~"^<[^@<>\\s]+@[^:<>\\s]+:[^<>\\r\\n]+>$")
#Source:              #Text & !~"^<.*>$" & !~"@pending:" & !~"^unknown:"

#ContractShapeRef:  string & =~"^ref:[A-Za-z_][A-Za-z0-9_]*\\.shape$"
#ContractDTypeRef:  string & =~"^ref:[A-Za-z_][A-Za-z0-9_]*\\.dtype$"
#Binding:           #Text & =~"^(ref|bind):\\S.*$"
#ShapeRef:          #Binding & =~"\\.shape$"
#DTypeRef:          #Binding & =~"\\.dtype$"
#ResolvedDTypeRef:  #DTypeRef & =~"^ref:"
#Unknown:           #Text & =~"^unknown:\\s*\\S.*$"
#UpstreamReference: #Text & !~"^(unknown:|<)"
#ReferenceList: [#UpstreamReference, ...#UpstreamReference]

#ReferenceOrList: #UpstreamReference | #ReferenceList

// Contract: algorithm and hardware
// --------------------------------

// schema_docs/1: Contract is complete; unknown/bind are not allowed here.
#AlgorithmRequirement: {
	constant?: {
		[#Identifier]: {
			value!:   number | #KnownValueText
			meaning!: #Text
		}
	} & #NonemptyMap
	input!: {
		[#Identifier]: #Input
	} & #NonemptyMap
	output!: {
		[#Identifier]: #Output
	} & #NonemptyMap
	compute!: [#Compute, ...#Compute]
	golden_reference!: {
		source!: #ReferenceEntry
		tolerance!: {
			[#Identifier]: #Text
		} & #NonemptyMap
		gap?: #Text
	}

	// Key equality is checked, not inferred from prose or a sample's outputs.
	for name, _ in output {
		golden_reference: tolerance: "\(name)"!: #Text
	}
	for name, _ in golden_reference.tolerance {
		output: "\(name)"!: #Output
	}
}

#Output: {
	meaning!: #Text
	shape!:   #Shape
	dtype!:   #DType
}

#Input: {
	#Output
	range?: [number | #Text, number | #Text] | #Text
	default?: #Scalar
}

#Compute: {
	name!: #Text
	math!: #Text
	symbol?: {
		[#Identifier]: {
			meaning!: #Text
			shape!:   #Shape | #ContractShapeRef | "not_applicable"
			dtype!:   #DType | #ContractDTypeRef | "not_applicable"
		}
	} & #NonemptyMap
}

// schema_docs/2: execution and feature are separate vocabularies. A shared
// spelling across them is harmless; hardware names can contain dots/spaces/"/".
#HardwareSpecification: {
	device!: #HardwareFact
	arch!:   #HardwareFact
	execution!: {
		[#Text]: #HardwareFact
	} & #NonemptyMap
	feature!: {
		[#Text]: #HardwareFact
	} & #NonemptyMap
	stack!: {
		language!:  #HardwareFact
		template?:  #HardwareFact
		toolchain!: #HardwareFact
	}
	wiki!: {
		hardware!: #WikiReference
		programming!: {
			language!:    #WikiReference
			template?:    #WikiReference
			virtual_isa?: #WikiReference
			machine_isa?: #WikiReference
		}
		microbench!: #WikiReference
		sota!: {
			kernel!: #WikiReference
			schema!: #WikiReference
		}
		profile!: #WikiReference
	}
	for field, _ in stack if field == "template" {
		wiki: programming: template!: #WikiReference
	}
	for field, _ in wiki.programming if field == "template" {
		stack: template!: #Text
	}
}

// Design: mathematical rewrite
// ----------------------------

// schema_docs/3: mathematical definitions, logical lifetime, and delta ledger.
#MathReshape: M={
	thesis!: #Text
	rewrite!: [#Rewrite, ...#Rewrite]
	ledger?: #Ledger
	let Steps = [for r in rewrite {r.step}]
	if !list.UniqueItems(Steps) {
		rewrite: error("rewrite step IDs must be unique")
	}

	// Inspect keys for presence. Testing the value against bottom could swallow
	// an error introduced by the constraint inside the conditional itself.
	for field, entries in M if field == "ledger" {
		for resource, rows in entries if resource != "baseline" {
			for row in rows for step in row.by {
				if !list.Contains(Steps, step) {
					ledger: error("ledger.by references unknown rewrite step \(step)")
				}
			}
		}
	}
}

#Rewrite: {
	step!: #Text
	name!: #Text
	math!: #Text
	symbol?: {
		[#Identifier]: #MathSymbol
	} & #NonemptyMap
	arithmetic?: {
		approx?:     #Text
		accumulate?: #Text
	} & #NonemptyMap
}

#MathSymbol: S={
	meaning!: #Text
	shape?:   #Shape | #ShapeRef | #Unknown
	dtype?:   #DType | #DTypeRef | #Unknown
	life?:    #Text & =~"^(carried|stored)\\s+\\S.*$"
	binding?: #Binding
	value?:   #Literal
	for field, _ in S if field == "binding" {
		value?: _|_ // One definition owner; a binding is not another constant.
	}
	for field, _ in S if field == "life" {
		shape!: #Shape | #ShapeRef | #Unknown
		dtype!: #DType | #DTypeRef | #Unknown
	}
}

#Ledger: {
	baseline!: #Text
	matmul?: [#LedgerEntry, ...#LedgerEntry]
	vector?: [#LedgerEntry, ...#LedgerEntry]
	transcendental?: [#LedgerEntry, ...#LedgerEntry]
	traffic?: [#LedgerEntry, ...#LedgerEntry]
	footprint?: [#LedgerEntry, ...#LedgerEntry]
	depth?: [#LedgerEntry, ...#LedgerEntry]
	pass?: [#LedgerEntry, ...#LedgerEntry]
	comm?: [#LedgerEntry, ...#LedgerEntry]
	error?: [#LedgerEntry, ...#LedgerEntry]
} & struct.MinFields(2) // The baseline and at least one changed resource.

#LedgerEntry: {
	item!:   #Text
	change!: #Text & !~"unknown:"
	by!: [#Text, ...#Text] & list.UniqueItems
}

// Design: work organization
// -------------------------

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
	let Consumer = #PhaseID & !=producer
	consumer!: [Consumer, ...Consumer] & list.UniqueItems
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
		if refill == "none" {
			preassigned!: #Preassigned
		}
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

// Design: tile execution
// ----------------------

// schema_docs/5: finite nested pipe blocks; no arbitrary depth ceiling.
#TilePipeline: {
	thesis!: #Text
	phase!: {
		[#PhaseID]: #PipelinePhase
	} & #NonemptyMap
}
#PipelinePhase: {
	domain?: {
		[#Identifier]: #Domain
	} & #NonemptyMap
	milestone?: {
		// Built-in launch events cannot be redefined as local milestones.
		[string & =~"^[a-z_]+(\\([^\\r\\n]+\\))?$" & !~"^(kernel_done|launch_ready)(\\(|$)"]: #Events
	} & #NonemptyMap
	resource?: {
		[#Identifier]: #Resource
	} & #NonemptyMap
	lane!: {
		[#Identifier]: #Lane
	} & #NonemptyMap
}

#Resource: R={
	space!: #Text
	shape!: #Shape
	dtype!: #DType | #ResolvedDTypeRef
	slot?:  #PositiveExtent
	member?: #NonnegativeInt | ([#NonnegativeInt, ...#NonnegativeInt] & list.UniqueItems)
	view?: {
		[#ValuePattern]: #View
	} & #NonemptyMap
	for field, count in R if field == "slot" {
		if (count & string) == _|_ {
			if count > 1 {
				view!: {
					[#ValuePattern]: #View & {slot!: #Extent}
				} & #NonemptyMap
			}
		}
	}
	for field, views in R if field == "view" {
		for name, v in views for key, index in v if key == "slot" {
			if (index & string) == _|_ {
				if !list.Contains([for k, _ in R {k}], "slot") {
					view: "\(name)": slot: 0
				}
				for key, count in R if key == "slot" {
					if (count & string) == _|_ {
						view: "\(name)": slot: <count
					}
				}
			}
		}
	}
}

#ValuePattern: string & =~"^[A-Za-z_][A-Za-z0-9_]*(\\([^\\r\\n]+\\))?$"

// {} names a whole-slot view with inherited dtype in single-slot storage.
#View: {
	slot?: #Extent
	part?: {
		[#Identifier]: _#Region
	} & #NonemptyMap
	dtype?: #DType | #ResolvedDTypeRef
}

#Lane: {
	info!: {
		member!:   #MemberSelector
		mode?:     #Mode
		register?: #PositiveInt
		verb?: {
			[#Identifier]: #Text | {
				meaning!: #Text
				mode!:    #Mode
			}
		} & #NonemptyMap
	}
	pipe!: #Pipe
	edge?: {
		data?:  #Edges
		reuse?: #Edges
		sync?:  #Edges
	} & #NonemptyMap
}
#MemberSelector: #Text & =~"^[^\\[\\]\\r\\n]+\\[[0-9]+(:[0-9]+)?\\](\\.[^\\[\\]\\r\\n]+\\[[0-9]+(:[0-9]+)?\\])*$"

#Pipe: P={
	repeat?:   #Repeat
	prologue?: #Actions
	steady?:   #Actions
	epilogue?: #Actions
	for field, _ in P if field == "repeat" {steady!: #Actions}
} & #NonemptyMap
#Actions: [#Action | #Pipe, ...#Action | #Pipe]
#Action: #Text & =~"^[A-Za-z_][A-Za-z0-9_]*:[A-Za-z_][^\\r\\n]*$"
#Events: [#EventReference, ...#EventReference]
#EventReference: #Text & =~"^([A-Za-z_][A-Za-z0-9_]*(\\[(\\*|[0-9]+)\\])?/)?[A-Za-z_][A-Za-z0-9_]*:[A-Za-z_][^\\r\\n]*$"
#Edges: {
	[#EventReference]: #Events
} & #NonemptyMap

#Domain: #Range | "worktile" | {
	flatten!: [#Identifier, ...#Identifier] & list.UniqueItems
}

#Repeat: (string & =~"^[A-Za-z_][A-Za-z0-9_]*(\\[[^\\r\\n]+\\])?$") | ({
	[#Identifier]: #Domain
} & struct.MinFields(1) & struct.MaxFields(1))

// Realization: implementation choices and feedback
// ------------------------------------------------

// schema_docs/6: feedback is a proposal with evidence, only under prefer.
#Realization: {
	require?: [#ImplementationChoice, ...#ImplementationChoice]
	prefer?: [#Preference, ...#Preference]
	forbid?: [#ImplementationChoice, ...#ImplementationChoice]
} & #NonemptyMap
#ImplementationChoice: {
	item!:   #UpstreamReference | (#ReferenceList & list.UniqueItems)
	choice!: #Text
	reason!: #Text
}
#Preference: {
	#ImplementationChoice
	feedback?: {
		choice!:   #Text
		evidence!: #BoundReference
	}
}
