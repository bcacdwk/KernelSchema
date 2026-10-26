package kernel

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
