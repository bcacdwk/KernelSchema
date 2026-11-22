This is the shared K2 CPU contract fixture, not execution evidence.

`build.py` reads the K1 original files at the paths/hashes in `k1_inputs.json`
(material commit `35443d2f6cefe424e9adcd6d8d66dc17aa267d99`). It preserves
those bytes and emits real JCS/Ref/Source/Task/Schema objects into a new portable
bundle. Task criteria stay in K1; this fixture does not maintain another copy.
Its evaluator descriptor and profile are explicitly synthetic/interface-only.
No adapter, evaluator execution, model or GPU qualification is claimed.

With the source checkout installed editable:

```sh
python -B tests/workflows/fixtures/v1/row_softmax/build.py --out /tmp/ks-synthetic --operation implement
```

Use a new output directory per build. `design` and `optimize` select the other
request shapes. The three business APIs deliberately reject the fixture's
unimplemented `synthetic` backend; CPU tests use their own explicit fake owner.
`jcs_vectors.json` supplies independent expected canonical bytes and hashes,
including RFC 8785 UTF-16 ordering, control escaping and safe integer limits.
