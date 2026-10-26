package kernel

import (
	"list"
	"struct"
)

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
		[string & =~"^[a-z_]+(\\([^\\r\\n]+\\))?$"]: #Events
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
		[#Identifier]: #Range
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
