package kernel

import (
	"list"
	"struct"
)

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
