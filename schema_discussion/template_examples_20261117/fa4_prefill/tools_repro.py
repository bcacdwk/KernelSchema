"""Finite DOC5 window reproductions; no YAML or GPU mutation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from schema_template.tools.pipeline import build_step

def role(verb, noun, pipe, waiting=None):
    result = {'Info': {'Describe': noun, 'Member': 'warp[0:1]', 'Using_verb': {verb: {noun: {'execution_mode': 'serial', 'describe': noun}}}}, 'Pipe': pipe}
    if waiting:
        result['Waiting'] = waiting
    return result

index = {'t': {'meaning': 'Assigned work', 'definition': 'worktile'},
         'i': {'meaning': 'Local iteration', 'definition': [0, 'tile_count(t)']},
         'n': {'meaning': 'Continuous iteration', 'definition': {'flatten': ['t', 'i']}}}
index['d'] = {'meaning': 'Half', 'definition': [0, 2]}
fixture = {'Index': index, 'Role': {
    'LOAD': role('load', 'k', [{'for(t)': [{'if(valid(t))': [{'for(n[t])': [{'for(d)': ['load:k(n,d)']}]}]}]}]),
    'USE': role('compute', 'k', [{'for(t)': [{'if(valid(t))': [{'for(d)': ['compute:k(n[t,0],d)']}, {'for(n[t,1:])': [{'for(d)': ['compute:k(n,d)']}]}]}]}],
                 {'Released': ['{issue compute:k(n+2,0)} ← [{done LOAD::load:k(n,0)}]']})}}
try:
    graph = build_step('MAIN', fixture, {'unit': 'cta', 'unit_count': 1}, {'tile_count(t)': 4, 'valid(t)': 1}, worktiles=1)
    print('flatten_boundary: unexpectedly completed')
except Exception as error:
    print('flatten_boundary:', type(error).__name__, str(error))

fixture = {'Index': index, 'Role': {
    'USE': role('compute', 'k', [{'for(t)': [{'if(valid(t))': [{'for(n[t,1:])': ['compute:k(n-1,0)']}, 'compute:k(n[t,tile_count(t)-1],0)']}]}],
                {'Prepared': ['{issue compute:k(n[t,i+1],0)} ← [{done LOAD::load:k(n[t,i])}]']}),
    'LOAD': role('load', 'k', [{'for(t)': [{'if(valid(t))': [{'for(n[t])': ['load:k(n)']}]}]}])}}
try:
    graph = build_step('MAIN', fixture, {'unit': 'cta', 'unit_count': 1}, {'tile_count(0)': 1, 'tile_count(1)': 0, 'valid(0)': 1, 'valid(1)': 0}, worktiles=2)
    print('padding_window: unexpectedly completed')
except Exception as error:
    print('padding_window:', type(error).__name__, str(error))
