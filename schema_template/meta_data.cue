package kernel

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
