package kernel

import "list"

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
