package kernel

import "struct"

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
#MemberRegion: #Range | [#Range, ...#Range]

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
