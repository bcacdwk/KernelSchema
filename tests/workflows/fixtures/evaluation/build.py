"""Actual evaluator bundles for CPU logic tests; no GPU qualification."""
from kernelschema.artifacts import Publication, Bundle
from kernelschema.evaluation.task import freeze_task, stage
from kernelschema.resources import resource


def build(root, source=None):
    publication = Publication(root)
    environment = publication.put_bytes("environment.json", b'{"qualification":"none","mode":"CPU logic fixture"}')
    refs = freeze_task(publication, environment=environment)
    if source is None:
        source = resource("examples/workflows/row_softmax/evaluator/baseline.py").read_bytes()
    candidate = publication.put_source("candidate", source)
    publication.commit([refs["task"], candidate])
    return {**refs, "candidate": candidate, "bundle": Bundle(root)}


def staged(root, source=None):
    fixture = build(root / "bundle", source)
    snapshot = stage(fixture["bundle"], fixture["task"], fixture["candidate"], root / "stage")
    return fixture, snapshot
