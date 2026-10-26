package kernel

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
