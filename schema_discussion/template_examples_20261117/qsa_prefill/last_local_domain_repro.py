"""Read-only reproduction: a bounded local domain depends on the current worktile."""
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'schema_template'))
from tools.pipeline import build_step, GraphError

step = {
    'Index': {
        't': {'meaning': 'Accepted local work ordinal', 'definition': 'worktile'},
        'i': {'meaning': 'Bounded local iteration', 'definition': [0, 'tile_count(t)']},
    },
    'Role': {'MAIN': {
        'Info': {'Describe': 'Compute and store a local result', 'Member': 'warp[0:1]',
                 'Using_verb': {
                     'compute': {'x': {'execution_mode': 'serial', 'describe': 'Compute local x'}},
                     'store': {'y': {'execution_mode': 'serial', 'describe': 'Store final local x'}},
                 }},
        'Pipe': [{'for(t)': [{'for(i)': ['compute:x(t,i)']}, 'store:y(t)']}],
        'Waiting': {'Prepared': ['{issue store:y(t)} ← [{done compute:x(t,last(i))}]']},
    }},
}
for label, replacement in [('bounded last(i)', 'last(i)'), ('equivalent endpoint', 'tile_count(t)-1')]:
    current = deepcopy(step)
    current['Role']['MAIN']['Waiting']['Prepared'][0] = current['Role']['MAIN']['Waiting']['Prepared'][0].replace('last(i)', replacement)
    try:
        graph = build_step('MAIN', current, {'unit': 'cta', 'unit_count': 1},
                           parameters={'tile_count(t)': 2}, worktiles=1, dynamic_worktiles=True)
        print(label, len(graph.operations), 'actions', 'acyclic=', not graph.cycle)
    except GraphError as error:
        print(label, type(error).__name__, str(error))
