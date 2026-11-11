#!/usr/bin/env python3
"""Static pipe census. Reads the selected corpus; writes only pipe_grammar/data.

A: unique YAML node identities per source file, reached from local roles/pipes.
B: local roles/pipes with YAML aliases expanded at their occurrence positions.
C: each selected node's effective area, resolving direct-parent area references.
No repeat range is executed or multiplied. Exact groups compare parsed ordered
pipe structure, including action strings, but exclude surrounding role/domain
contracts: they do not assert semantic equivalence.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import yaml
from yaml.events import (AliasEvent, MappingStartEvent, MappingEndEvent,
                         SequenceStartEvent, SequenceEndEvent, ScalarEvent)
from yaml.nodes import ScalarNode

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / 'schema_extract/archive/20261109-ten-operator-text-audit'
OUT = ROOT / 'schema_discussion/pipe_grammar/data'
AREAS = ('tile_pipeline', 'work_schedule', 'math_reshape')
REGIONS = ('prologue', 'steady', 'epilogue')
VERSION = 'kernelschema.pipe_census.v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


@dataclass(eq=False)
class Node:
    kind: str
    value: object
    line: int
    start: int
    end: int = 0
    anchor: str | None = None
    tag: str | None = None


@dataclass
class Edge:
    node: Node
    line: int
    alias: str | None = None


SCALAR_LOADER = yaml.SafeLoader('')
RESOLVER = yaml.resolver.Resolver()


def scalar(event):
    tag = event.tag or RESOLVER.resolve(ScalarNode, event.value, event.implicit)
    n = ScalarNode(tag, event.value)
    constructor = SCALAR_LOADER.yaml_constructors.get(tag)
    return (constructor(SCALAR_LOADER, n) if constructor else event.value), tag


class Source:
    def __init__(self, path):
        self.path = path
        self.text = (ROOT / path).read_text()
        self.issues = []
        self.aliases = []
        self.anchors = {}
        self.events = iter(yaml.parse(self.text, Loader=yaml.CSafeLoader))
        next(self.events)  # stream start
        next(self.events)  # document start
        self.root = self.parse(next(self.events))
        self.check_cycles(self.root.node, set(), set())

    def parse(self, e):
        if isinstance(e, AliasEvent):
            if e.anchor not in self.anchors:
                raise ValueError(f'undefined alias {e.anchor} at line {e.start_mark.line + 1}')
            self.aliases.append({'anchor': e.anchor, 'line': e.start_mark.line + 1,
                                 'declaration_line': self.anchors[e.anchor].line})
            return Edge(self.anchors[e.anchor], e.start_mark.line + 1, e.anchor)
        if isinstance(e, ScalarEvent):
            value, tag = scalar(e)
            n = Node('scalar', value, e.start_mark.line + 1, e.start_mark.index,
                     e.end_mark.index, e.anchor, tag)
        elif isinstance(e, MappingStartEvent):
            n = Node('map', [], e.start_mark.line + 1, e.start_mark.index, anchor=e.anchor)
        elif isinstance(e, SequenceStartEvent):
            n = Node('seq', [], e.start_mark.line + 1, e.start_mark.index, anchor=e.anchor)
        else:
            raise ValueError(f'unexpected YAML event {type(e).__name__}')
        if e.anchor:
            if e.anchor in self.anchors:
                self.issues.append({'kind': 'duplicate_anchor', 'anchor': e.anchor,
                                    'line': n.line, 'previous_line': self.anchors[e.anchor].line})
            self.anchors[e.anchor] = n
        if n.kind == 'map':
            keys = set()
            while not isinstance((event := next(self.events)), MappingEndEvent):
                k = self.parse(event)
                v = self.parse(next(self.events))
                key = str(k.node.value)
                if key in keys:
                    self.issues.append({'kind': 'duplicate_mapping_key', 'key': key, 'line': k.line})
                keys.add(key)
                n.value.append((k, v))
            n.end = event.end_mark.index
        elif n.kind == 'seq':
            while not isinstance((event := next(self.events)), SequenceEndEvent):
                n.value.append(self.parse(event))
            n.end = event.end_mark.index
        return Edge(n, n.line)

    def check_cycles(self, n, active, done):
        if id(n) in active:
            self.issues.append({'kind': 'recursive_anchor', 'anchor': n.anchor, 'line': n.line})
            return
        if id(n) in done:
            return
        active = active | {id(n)}
        for e in children(n):
            self.check_cycles(e.node, active, done)
        done.add(id(n))


def children(n):
    if n.kind == 'map':
        return [v for pair in n.value for v in pair]
    return n.value if n.kind == 'seq' else []


def pairs(n):
    return [(str(k.node.value), v) for k, v in n.value] if n and n.kind == 'map' else []


def get(n, key):
    return next((v for k, v in pairs(n) if k == key), None)


def at(n, *keys):
    for key in keys:
        e = get(n, key)
        if e is None:
            return None
        n = e.node
    return n


def plain(n, active=None):
    if n is None:
        return None
    active = active or set()
    if id(n) in active:
        return {'$recursive_anchor': n.anchor}
    active = active | {id(n)}
    if n.kind == 'map':
        return {k: plain(v.node, active) for k, v in pairs(n)}
    if n.kind == 'seq':
        return [plain(v.node, active) for v in n.value]
    return n.value


def location(source, n, path, edge=None):
    result = {'file': source.path, 'path': path, 'line': n.line}
    if edge and edge.alias:
        result.update(occurrence_line=edge.line, alias=edge.alias)
    return result


def roles(area):
    for phase, pe in pairs(at(area, 'phase')):
        for role, re_ in pairs(at(pe.node, 'role')):
            yield phase, pe, role, re_


def new_stats():
    return {'roles': 0, 'pipe_blocks': 0, 'root_blocks': 0, 'nested_blocks': 0,
            'static_actions': 0, 'max_depth': 0, 'depth_counts': Counter(),
            'field_combinations': Counter(), 'repeat_forms': Counter(),
            'repeat_binding_flags': Counter(), 'repeat_literals': Counter(),
            'region_presence': Counter(), 'region_actions': Counter(),
            'region_subblocks': Counter(), 'pure_repeat_steady_subblocks': 0,
            'no_repeat_steady_only_blocks': 0, 'steady_only_containers_with_subblocks': 0,
            'steady_only_containers_only_subblocks': 0,
            'loops_immediately_in_prologue': 0, 'loops_immediately_in_epilogue': 0,
            'loops_anywhere_under_prologue': 0, 'loops_anywhere_under_epilogue': 0,
            'lists_with_adjacent_loops': 0, 'adjacent_loop_pairs': 0,
            'lists_with_mixed_actions_and_loops': 0,
            'lists_with_mixed_actions_and_subblocks': 0,
            'invalid_pipe_items': 0}


def add_stats(target, source):
    for k, v in source.items():
        if isinstance(v, (dict, Counter)):
            target[k].update(v)
        elif k == 'max_depth':
            target[k] = max(target[k], v)
        else:
            target[k] += v


def frozen_stats(s):
    return {k: dict(sorted(v.items())) if isinstance(v, (dict, Counter)) else v
            for k, v in s.items()}


def binding_flags(value):
    flags = set()
    if value == 'worktile':
        flags.add('worktile')
    if isinstance(value, dict):
        if 'flatten' in value:
            flags.add('flatten')
        for k, v in value.items():
            if k != 'flatten':
                flags |= binding_flags(v)
    if isinstance(value, list):
        if len(value) == 2:
            if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                flags.add('fixed_numeric_bounds')
            elif any(isinstance(v, str) for v in value):
                flags.add('symbolic_bounds')
        for v in value:
            flags |= binding_flags(v)
    return flags


def repeat_detail(n, domains, source, path):
    if n is None:
        return {'form': 'absent'}
    value = plain(n)
    out = {'value': value, 'source_literal': source.text[n.start:n.end]}
    if isinstance(value, dict):
        out['form'] = 'inline_local_domain'
        out['definition'] = value
    elif isinstance(value, str):
        out['form'] = 'domain_slice' if '[' in value else 'domain_name'
        name = value.split('[', 1)[0]
        out['domain_name'] = name
        de = get(domains, name)
        if de:
            out['definition'] = plain(de.node)
            out['definition_ref'] = location(source, de.node, path + '.' + name, de)
        else:
            out['binding_unresolved'] = True
    else:
        out['form'] = 'other'
    flags = binding_flags(out.get('definition'))
    definition = out.get('definition')
    if isinstance(definition, dict) and 'flatten' in definition:
        out['flatten_components'] = {}
        for component in definition['flatten']:
            component_name = str(component).split('[', 1)[0]
            ce = get(domains, component_name)
            if ce:
                cv = plain(ce.node)
                out['flatten_components'][component_name] = cv
                flags |= binding_flags(cv)
    out['binding_flags'] = sorted(flags)
    return out


def inspect_pipe(pipe_edge, source, path, domains, phase, seen=None, include_trace=True,
                 initial_aliases=None):
    """Dedup only contributions in A; still traverse aliases to find unseen nodes."""
    s = new_stats()
    blocks, actions = [], []
    seen = seen if seen is not None else {'block': set(), 'action': set()}
    dedup = bool(seen.get('dedup'))

    def walk(edge, p, depth, region, ancestors, active, aliases):
        n = edge.node
        alias_chain = aliases + ([{'anchor': edge.alias, 'line': edge.line,
                                   'declaration_line': n.line}] if edge.alias else [])
        if id(n) in active:
            s['invalid_pipe_items'] += 1
            return
        if n.kind != 'map':
            s['invalid_pipe_items'] += 1
            return
        first = id(n) not in seen['block'] or not dedup
        seen['block'].add(id(n))
        fields = [k for k, _ in pairs(n)]
        rn = at(n, 'repeat')
        detail = repeat_detail(rn, domains, source, f'design.tile_pipeline.phase.{phase}.domain')
        sections = {}
        for section in REGIONS:
            seq = at(n, section)
            if seq is None:
                continue
            if seq.kind != 'seq':
                if first:
                    s['invalid_pipe_items'] += 1
                sections[section] = {'invalid_type': seq.kind}
                continue
            kinds = ['loop' if e.node.kind == 'map' and get(e.node, 'repeat') else
                     'block' if e.node.kind == 'map' else
                     'action' if e.node.kind == 'scalar' and isinstance(e.node.value, str) else
                     'invalid' for e in seq.value]
            actions_n, blocks_n = kinds.count('action'), kinds.count('loop') + kinds.count('block')
            adjacent = sum(a == b == 'loop' for a, b in zip(kinds, kinds[1:]))
            sections[section] = {'actions': actions_n, 'subblocks': blocks_n,
                                 'loops': kinds.count('loop'), 'item_kinds': kinds,
                                 'adjacent_loop_pairs': adjacent}
            region_edge = get(n, section)
            if region_edge.alias:
                sections[section]['alias'] = {'anchor': region_edge.alias,
                                               'line': region_edge.line,
                                               'declaration_line': seq.line}
            if first:
                s['region_presence'][section] += 1
                s['region_actions'][section] += actions_n
                s['region_subblocks'][section] += blocks_n
                s['adjacent_loop_pairs'] += adjacent
                s['lists_with_adjacent_loops'] += bool(adjacent)
                s['lists_with_mixed_actions_and_loops'] += bool(actions_n and 'loop' in kinds)
                s['lists_with_mixed_actions_and_subblocks'] += bool(actions_n and blocks_n)
        if first:
            s['pipe_blocks'] += 1
            s['root_blocks' if depth == 1 else 'nested_blocks'] += 1
            s['max_depth'] = max(s['max_depth'], depth)
            s['depth_counts'][str(depth)] += 1
            s['field_combinations']['+'.join(sorted(fields))] += 1
            s['repeat_forms'][detail['form']] += 1
            if rn:
                s['repeat_literals'][compact(detail['value'])] += 1
                s['repeat_binding_flags'].update(detail['binding_flags'])
                for section in ('prologue', 'epilogue'):
                    s['loops_immediately_in_' + section] += region == section
                    s['loops_anywhere_under_' + section] += section in ancestors
            s['pure_repeat_steady_subblocks'] += depth > 1 and set(fields) == {'repeat', 'steady'}
            if set(fields) == {'steady'}:
                s['no_repeat_steady_only_blocks'] += 1
                st = sections.get('steady', {})
                s['steady_only_containers_with_subblocks'] += bool(st.get('subblocks'))
                s['steady_only_containers_only_subblocks'] += bool(st.get('subblocks')) and not st.get('actions')
            if include_trace:
                trace = {'path': p, 'line': n.line, 'depth': depth, 'parent_region': region,
                         'source_node_id': f'map@{n.start}',
                         'fields': fields, 'repeat': detail, 'regions': sections}
                if alias_chain:
                    trace['alias_chain'] = alias_chain
                    trace['occurrence_line'] = alias_chain[0]['line']
                blocks.append(trace)
        for section in REGIONS:
            seq = at(n, section)
            if not seq or seq.kind != 'seq':
                continue
            region_edge = get(n, section)
            region_aliases = alias_chain + ([{'anchor': region_edge.alias,
                                              'line': region_edge.line,
                                              'declaration_line': seq.line}]
                                            if region_edge.alias else [])
            for i, item in enumerate(seq.value):
                ip = f'{p}.{section}[{i}]'
                if item.node.kind == 'map':
                    walk(item, ip, depth + 1, section, ancestors + [section],
                         active | {id(n)}, region_aliases)
                elif item.node.kind == 'scalar' and isinstance(item.node.value, str):
                    if not dedup or id(item.node) not in seen['action']:
                        s['static_actions'] += 1
                        if include_trace:
                            a = {'path': ip, 'line': item.node.line, 'action': item.node.value,
                                 'source_node_id': f'scalar@{item.node.start}'}
                            if item.alias or region_aliases:
                                a['alias_chain'] = region_aliases + ([{'anchor': item.alias, 'line': item.line,
                                                                  'declaration_line': item.node.line}] if item.alias else [])
                            actions.append(a)
                    seen['action'].add(id(item.node))
                elif first:
                    s['invalid_pipe_items'] += 1
    walk(pipe_edge, path, 1, None, [], set(), initial_aliases or [])
    return s, blocks, actions


def shape_pipe(p):
    if not isinstance(p, dict):
        return p
    return {k: [shape_pipe(v) if isinstance(v, dict) else '<action>' for v in val]
            if k in REGIONS and isinstance(val, list) else val for k, val in p.items()}


def process_operator(op_id, original):
    records, sources, issues = {}, {}, []
    for entry in original['nodes']:
        node_id = entry['node']
        rec = {k: entry.get(k) for k in ('node', 'parent', 'category', 'authored',
                                       'material_present', 'semantic_eligible', 'in_current_plan')}
        rec['manifest_parent_raw'] = entry.get('parent')
        rec['parent'] = entry.get('parent') or None
        rec['is_root'] = rec['parent'] is None
        rec['included'] = bool(entry.get('semantic_eligible'))
        if rec['included']:
            path = entry['source_file'] if original['group'] == 'template' else entry['snapshot']
            rec.update(file=path, sha256=sha((ROOT / path).read_bytes()),
                       frozen_snapshot=entry['snapshot'], frozen_sha256=entry['sha256'])
            rec['frozen_snapshot_actual_sha256'] = sha((ROOT / entry['snapshot']).read_bytes())
            rec['matches_frozen'] = rec['sha256'] == entry['sha256'] == rec['frozen_snapshot_actual_sha256']
            try:
                source = Source(path)
                rec['yaml_anchors'] = len(source.anchors)
                rec['yaml_aliases'] = len(source.aliases)
                rec['yaml_issues'] = source.issues
                rec['yaml_valid'] = not source.issues
                if rec['yaml_valid']:
                    sources[node_id] = source
                issues.extend(dict(i, node=node_id) for i in source.issues)
            except Exception as exc:
                rec['yaml_valid'] = False
                issues.append({'kind': 'yaml_parse_error', 'node': node_id, 'error': str(exc)})
        records[node_id] = rec
    resolved = {}

    def resolve(node_id, area, active=()):
        key = (node_id, area)
        if key in resolved:
            return resolved[key]
        if node_id in active:
            raise ValueError(f'parent reference cycle {active} -> {node_id} ({area})')
        if node_id not in sources:
            raise ValueError(f'provider missing from selected material: {node_id} ({area})')
        area_node = at(sources[node_id].root.node, 'design', area)
        parent_ref = get(area_node, 'from_parent')
        if parent_ref:
            parent = plain(parent_ref.node)
            if parent != records[node_id]['parent']:
                raise ValueError(f'from_parent {parent!r} differs from direct parent {records[node_id]["parent"]!r}')
            if len(pairs(area_node)) != 1:
                raise ValueError('from_parent area also contains local keys')
            provider, chain, n = resolve(parent, area, active + (node_id,))
            value = provider, [node_id] + chain, n
        else:
            value = node_id, [node_id], area_node
        resolved[key] = value
        return value

    for node_id, src in sources.items():
        rec = records[node_id]
        rec['areas'] = {}
        actual_parent = plain(at(src.root.node, 'meta_data', 'dispatch', 'parent'))
        if actual_parent != rec['parent']:
            issues.append({'kind': 'manifest_parent_mismatch', 'node': node_id,
                           'manifest_parent': rec['parent'], 'yaml_parent': actual_parent})
        for area in AREAS:
            try:
                provider, chain, an = resolve(node_id, area)
                rec['areas'][area] = {'provider': provider, 'chain': chain,
                                     'local': len(chain) == 1,
                                     'ref': location(sources[provider], an, 'design.' + area) if an else None}
            except ValueError as exc:
                issues.append({'kind': 'inheritance_error', 'node': node_id, 'area': area, 'error': str(exc)})

    groups = {}
    role_contracts = {}
    contexts = {}
    totals = {scope: new_stats() for scope in ('A_decl', 'B_local', 'C_effective')}
    roots = {scope: new_stats() for scope in totals}
    branches = {scope: new_stats() for scope in totals}
    provider_nodes = Counter()

    for node_id, source in sources.items():
        rec = records[node_id]
        per = {scope: new_stats() for scope in totals}
        rec['pipe_occurrences'] = []
        local = at(source.root.node, 'design', 'tile_pipeline')
        seen = {'role': set(), 'block': set(), 'action': set(), 'dedup': True}
        for phase, pe, role, re_ in roles(local):
            pipe = get(re_.node, 'pipe')
            if id(re_.node) not in seen['role']:
                per['A_decl']['roles'] += 1
                seen['role'].add(id(re_.node))
            if pipe:
                stats, _, _ = inspect_pipe(pipe, source, f'design.tile_pipeline.phase.{phase}.role.{role}.pipe',
                                           at(pe.node, 'domain'), phase, seen, False)
                add_stats(per['A_decl'], stats)
        if (node_id, 'tile_pipeline') not in resolved:
            continue
        provider, chain, effective = resolve(node_id, 'tile_pipeline')
        provider_nodes[provider] += 1
        ps = sources[provider]
        rec['local_pipeline_roles'] = len(list(roles(local)))
        rec['effective_pipeline_roles'] = len(list(roles(effective)))
        rec['inherited_pipeline_roles'] = rec['effective_pipeline_roles'] if provider != node_id else 0
        for phase, pe, role, re_ in roles(effective):
            pipe = get(re_.node, 'pipe')
            scopes = ['C_effective'] + (['B_local'] if provider == node_id else [])
            for scope in scopes:
                per[scope]['roles'] += 1
            if not pipe:
                issues.append({'kind': 'role_without_pipe', 'node': node_id, 'phase': phase, 'role': role})
                continue
            path = f'design.tile_pipeline.phase.{phase}.role.{role}.pipe'
            inherited_aliases = [{'anchor': e.alias, 'line': e.line,
                                  'declaration_line': e.node.line} for e in (pe, re_) if e.alias]
            stats, blocks, actions = inspect_pipe(pipe, ps, path, at(pe.node, 'domain'), phase,
                                                  initial_aliases=inherited_aliases)
            for scope in scopes:
                add_stats(per[scope], stats)
            canonical = plain(pipe.node)
            fingerprint = sha(compact(canonical).encode())
            group_id = f'{op_id}:pipe:{fingerprint[:16]}'
            context_id = f'{node_id}:{phase}'
            if context_id not in contexts:
                context = {'node': node_id, 'phase': phase, 'pipeline_provider': provider,
                           'pipeline_chain': chain,
                           'domain': plain(at(pe.node, 'domain')) or {},
                           'domain_ref': location(ps, at(pe.node, 'domain'), f'design.tile_pipeline.phase.{phase}.domain')
                           if at(pe.node, 'domain') else None,
                           'resolved_areas_ref': {'file_record': node_id, 'field': 'areas'}}
                if (node_id, 'work_schedule') in resolved:
                    sp, _, sn = resolve(node_id, 'work_schedule')
                    phase_node = at(sn, 'phase', phase)
                    context['schedule_phase_ref'] = location(sources[sp], phase_node, f'design.work_schedule.phase.{phase}') if phase_node else None
                    if phase_node:
                        context['schedule_details'] = {k: plain(at(phase_node, k)) for k in ('mapping', 'schedule') if at(phase_node, k)}
                        decomp = at(phase_node, 'decomposition')
                        context['decomposition_refs'] = {k: location(sources[sp], v.node, f'design.work_schedule.phase.{phase}.decomposition.{k}', v) for k, v in pairs(decomp)}
                if (node_id, 'math_reshape') in resolved:
                    mp, _, mn = resolve(node_id, 'math_reshape')
                    rewrite = at(mn, 'rewrite')
                    if rewrite and rewrite.kind == 'seq':
                        if 'math_steps' not in rec:
                            rec['math_steps'] = [dict(location(sources[mp], e.node, f'design.math_reshape.rewrite[{i}]'), step=plain(at(e.node, 'step')), name=plain(at(e.node, 'name'))) for i, e in enumerate(rewrite.value)]
                        context['math_steps_ref'] = {'file_record': node_id, 'field': 'math_steps'}
                contexts[context_id] = context
            contracts = {'info': plain(at(re_.node, 'info')),
                         'waits_for': plain(at(re_.node, 'waits_for'))}
            contract_id = 'role_contract:' + sha(compact(contracts).encode())[:16]
            role_contracts.setdefault(contract_id, contracts)
            member = {'node': node_id, 'is_root': rec['is_root'], 'phase': phase, 'role': role,
                      'path': path, 'file': ps.path, 'line': pipe.node.line,
                      'provider': provider, 'provider_chain': chain, 'scopes': scopes,
                      'context_id': context_id, 'role_line': re_.node.line,
                      'role_source_node_id': f'map@{re_.node.start}',
                      'role_contract_id': contract_id,
                      'role_info_ref': location(ps, at(re_.node, 'info'), path.rsplit('.', 1)[0] + '.info') if at(re_.node, 'info') else None,
                      'waits_for_ref': location(ps, at(re_.node, 'waits_for'), path.rsplit('.', 1)[0] + '.waits_for') if at(re_.node, 'waits_for') else None}
            if pipe.alias or re_.alias or pe.alias:
                alias = pipe if pipe.alias else re_ if re_.alias else pe
                member.update(occurrence_line=alias.line, alias=alias.alias)
            if group_id not in groups:
                groups[group_id] = {'group_id': group_id, 'sha256': fingerprint,
                                    'shape_sha256': sha(compact(shape_pipe(canonical)).encode()),
                                    'canonical_pipe': canonical, 'metrics': dict(frozen_stats(stats), roles=1),
                                    'representative': {'node': node_id, 'phase': phase, 'role': role,
                                                       'provider': provider, 'context_id': context_id},
                                    'blocks': blocks, 'actions': actions, 'members': []}
            # Keep per-occurrence coordinates without replicating action payload.
            member['block_locations'] = [{'path': b['path'], 'line': b['line'],
                                           'source_node_id': b['source_node_id'],
                                           **({'occurrence_line': b['occurrence_line'], 'alias_chain': b['alias_chain']} if 'alias_chain' in b else {})} for b in blocks]
            groups[group_id]['members'].append(member)
            rec['pipe_occurrences'].append({'group_id': group_id, 'phase': phase, 'role': role,
                                            'context_id': context_id, 'line': pipe.node.line, 'scopes': scopes})
        rec['counts'] = {k: frozen_stats(v) for k, v in per.items()}
        for scope in totals:
            add_stats(totals[scope], per[scope])
            add_stats(roots[scope] if rec['is_root'] else branches[scope], per[scope])

    for group in groups.values():
        group['occurrence_counts'] = {scope: sum(scope in m['scopes'] for m in group['members']) for scope in ('B_local', 'C_effective')}
        group['nodes'] = sorted(set(m['node'] for m in group['members']))
        group['provider_nodes'] = sorted(set(m['provider'] for m in group['members']))
    summary = {'planned_files': len(records),
               'included_files': sum(r['included'] for r in records.values()),
               'excluded_files': sum(not r['included'] for r in records.values()),
               'root_files': sum(r['included'] and r['is_root'] for r in records.values()),
               'branch_files': sum(r['included'] and not r['is_root'] for r in records.values()),
               'status_counts': dict(Counter(r['category'] for r in records.values())),
               'included_status_counts': dict(Counter(r['category'] for r in records.values() if r['included'])),
               'pipeline_local_files': sum(r.get('areas', {}).get('tile_pipeline', {}).get('local', False) for r in records.values()),
               'pipeline_inherited_files': sum(not r.get('areas', {}).get('tile_pipeline', {}).get('local', True) for r in records.values()),
               'pipeline_provider_nodes': dict(provider_nodes),
               'distinct_pipeline_providers': len(provider_nodes),
               'exact_pipe_groups': len(groups),
               'shape_pipe_groups': len({g['shape_sha256'] for g in groups.values()}),
               'counts': {k: frozen_stats(v) for k, v in totals.items()},
               'root_counts': {k: frozen_stats(v) for k, v in roots.items()},
               'branch_counts': {k: frozen_stats(v) for k, v in branches.items()},
               'issue_count': len(issues)}
    return {'format': VERSION, 'operator': op_id, 'group': original['group'],
            'name': original['name'], 'summary': summary, 'files': records,
            'contexts': contexts, 'role_contracts': role_contracts,
            'exact_pipe_groups': list(groups.values()), 'issues': issues}


def independent_counts(result):
    """Independent LibYAML composition / safe construction, no custom AST reuse."""
    src_nodes, parsed = {}, {}
    for nid, f in result['files'].items():
        if f['included'] and f.get('yaml_valid') and 'counts' in f:
            text = (ROOT / f['file']).read_text()
            src_nodes[nid] = yaml.compose(text, Loader=yaml.CSafeLoader)
            parsed[nid] = yaml.load(text, Loader=yaml.CSafeLoader)
    def yget(n, k):
        return next((v for key, v in n.value if key.value == k), None) if isinstance(n, yaml.MappingNode) else None
    def yroles(n):
        phases = yget(n, 'phase')
        for _, p in phases.value if phases else []:
            rs = yget(p, 'role')
            for _, r in rs.value if rs else []:
                yield r
    def independent_provider(nid, active=()):
        if nid in active:
            raise ValueError('Independent inheritance traversal found a cycle')
        area = parsed[nid]['design']['tile_pipeline']
        if 'from_parent' not in area:
            return nid
        direct = parsed[nid].get('meta_data', {}).get('dispatch', {}).get('parent')
        if area['from_parent'] != direct:
            raise ValueError('Independent inheritance traversal found a non-parent reference')
        return independent_provider(direct, active + (nid,))
    def count_yml(n, counts, depth, seen=None):
        if n is None:
            return
        if seen is not None and id(n) in seen:
            return
        if seen is not None:
            seen.add(id(n))
        counts['pipe_blocks'] += 1
        counts['root_blocks' if depth == 1 else 'nested_blocks'] += 1
        counts['max_depth'] = max(depth, counts['max_depth'])
        counts['depth_counts'][str(depth)] += 1
        for r in REGIONS:
            seq = yget(n, r)
            for x in seq.value if seq else []:
                if isinstance(x, yaml.MappingNode):
                    count_yml(x, counts, depth + 1, seen)
                else:
                    if seen is None or id(x) not in seen:
                        counts['static_actions'] += 1
                    if seen is not None:
                        seen.add(id(x))
    checks = []
    basic = ('roles', 'pipe_blocks', 'root_blocks', 'nested_blocks', 'static_actions', 'max_depth', 'depth_counts')
    for nid, root in src_nodes.items():
        source = yget(yget(root, 'design'), 'tile_pipeline')
        for scope in ('A_decl', 'B_local', 'C_effective'):
            seen = set() if scope == 'A_decl' else None
            if scope == 'C_effective':
                provider = independent_provider(nid)
                expected_provider = result['files'][nid]['areas']['tile_pipeline']['provider']
                if provider != expected_provider:
                    checks.append({'node': nid, 'metric': 'provider', 'primary': expected_provider,
                                   'independent': provider})
                source = yget(yget(src_nodes[provider], 'design'), 'tile_pipeline')
            c = new_stats()
            role_seen = set()
            for r in yroles(source):
                if scope != 'A_decl' or id(r) not in role_seen:
                    c['roles'] += 1
                    role_seen.add(id(r))
                count_yml(yget(r, 'pipe'), c, 1, seen)
            expected = result['files'][nid]['counts'][scope]
            for k in basic:
                if c[k] != expected[k]:
                    checks.append({'node': nid, 'scope': scope, 'metric': k,
                                   'primary': expected[k], 'independent': c[k]})
    return {'method': 'Independent LibYAML compose node identities and separate traversal; seven basic metrics per file/scope.',
            'file_scope_metric_checks': len(src_nodes) * 3 * len(basic),
            'mismatches': checks, 'passed': not checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-independent-validation', action='store_true')
    args = parser.parse_args()
    manifest_path = AUDIT / 'data/manifest.json'
    original = json.loads(manifest_path.read_text())
    all_results = {}
    for op_id, op in original['operators'].items():
        result = process_operator(op_id, op)
        write_json(OUT / 'operators' / f'{op_id}.json', result)
        all_results[op_id] = result
        s = result['summary']
        print(f'{op_id}: {s["included_files"]} files; B={s["counts"]["B_local"]["roles"]} roles/{s["counts"]["B_local"]["pipe_blocks"]} blocks; C={s["counts"]["C_effective"]["roles"]} roles/{s["counts"]["C_effective"]["pipe_blocks"]} blocks; {s["exact_pipe_groups"]} exact groups', flush=True)
    manifest = {'format': VERSION, 'input_manifest': str(manifest_path.relative_to(ROOT)),
                'input_manifest_sha256': sha(manifest_path.read_bytes()),
                'script': str(Path(__file__).relative_to(ROOT)),
                'script_sha256': sha(Path(__file__).read_bytes()),
                'semantics_reference': str((AUDIT / 'rules/schema_docs/5.design.tile_pipeline.md').relative_to(ROOT)),
                'semantics_reference_sha256': sha((AUDIT / 'rules/schema_docs/5.design.tile_pipeline.md').read_bytes()),
                'selection': '21 current template files plus 172 semantic_eligible extraction snapshots; frozen copies are provenance, not additional samples.',
                'operators': {k: {'group': v['group'], 'files': [{key: value for key, value in f.items() if key in ('node', 'parent', 'manifest_parent_raw', 'category', 'authored', 'material_present', 'semantic_eligible', 'included', 'yaml_valid', 'file', 'sha256', 'frozen_snapshot', 'frozen_sha256', 'frozen_snapshot_actual_sha256', 'matches_frozen')} for f in v['files'].values()]} for k, v in all_results.items()}}
    write_json(OUT / 'manifest.json', manifest)
    summary = {'format': VERSION, 'operators': {k: v['summary'] for k, v in all_results.items()}, 'totals': {}, 'cohorts': {}}
    for label, results in [('all', list(all_results.values())), ('template', [v for v in all_results.values() if v['group'] == 'template']), ('extract', [v for v in all_results.values() if v['group'] != 'template'])]:
        agg = {'files': sum(v['summary']['included_files'] for v in results),
               'status_counts': dict(sum((Counter(v['summary']['status_counts']) for v in results), Counter())),
               'exact_pipe_groups': sum(v['summary']['exact_pipe_groups'] for v in results),
               'counts': {}, 'root_counts': {}, 'branch_counts': {}}
        for group in ('counts', 'root_counts', 'branch_counts'):
            for scope in ('A_decl', 'B_local', 'C_effective'):
                stat = new_stats()
                for result in results:
                    add_stats(stat, result['summary'][group][scope])
                agg[group][scope] = frozen_stats(stat)
        if label == 'all':
            summary['totals'] = agg
        else:
            summary['cohorts'][label] = agg
    write_json(OUT / 'summary.json', summary)
    csv_columns = ['operator', 'scope', 'files', 'roles', 'pipe_blocks', 'root_blocks', 'nested_blocks', 'static_actions', 'max_depth', 'exact_pipe_groups']
    with (OUT / 'operator_counts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for op_id, result in all_results.items():
            for scope, counts in result['summary']['counts'].items():
                writer.writerow(dict(operator=op_id, scope=scope, files=result['summary']['included_files'], exact_pipe_groups=result['summary']['exact_pipe_groups'], **{k: counts[k] for k in csv_columns if k in counts}))
    validations = {'format': VERSION, 'input_hash_mismatches': [], 'operator_checks': {},
                   'template_baseline': {'expected_B_roles': 123, 'expected_B_blocks': 310,
                                         'expected_B_depth_counts': {'1': 123, '2': 134, '3': 53}},
                   'issues': {k: v['issues'] for k, v in all_results.items() if v['issues']}}
    selected_records = [f for result in all_results.values() for f in result['files'].values() if f['included']]
    validations['corpus_boundaries'] = {
        'planned_records': sum(len(result['files']) for result in all_results.values()),
        'included_records': len(selected_records),
        'unique_selected_paths': len({f['file'] for f in selected_records}),
        'template_files': summary['cohorts']['template']['files'],
        'extract_files': summary['cohorts']['extract']['files'],
        'excluded_failed_skeletons': summary['totals']['status_counts'].get('attempted_without_authored_result', 0),
        'excluded_not_started': summary['totals']['status_counts'].get('not_started', 0),
        'selected_root_files': sum(f['is_root'] for f in selected_records),
        'selected_branch_files': sum(not f['is_root'] for f in selected_records)}
    validations['corpus_boundaries']['matches_expected_selection'] = (
        validations['corpus_boundaries'] == {'planned_records': 241, 'included_records': 193,
        'unique_selected_paths': 193, 'template_files': 21, 'extract_files': 172,
        'excluded_failed_skeletons': 5, 'excluded_not_started': 43,
        'selected_root_files': 10, 'selected_branch_files': 183})
    for k, result in all_results.items():
        validations['input_hash_mismatches'].extend(dict(operator=k, node=n, file=f['file']) for n, f in result['files'].items() if f.get('included') and not f.get('matches_frozen'))
        if not args.skip_independent_validation:
            validations['operator_checks'][k] = independent_counts(result)
    template_b = summary['cohorts']['template']['counts']['B_local']
    validations['template_baseline']['observed'] = {k: template_b[k] for k in ('roles', 'pipe_blocks', 'depth_counts')}
    validations['template_baseline']['matches'] = template_b['roles'] == 123 and template_b['pipe_blocks'] == 310 and template_b['depth_counts'] == {'1': 123, '2': 134, '3': 53}
    validations['passed'] = (not validations['issues'] and not validations['input_hash_mismatches']
                             and validations['corpus_boundaries']['matches_expected_selection']
                             and all(v['passed'] for v in validations['operator_checks'].values()))
    validations['independent_validation_run'] = not args.skip_independent_validation
    write_json(OUT / 'validation.json', validations)
    write_json(OUT / 'schema.json', {
        'format': VERSION,
        'counts': {'A_decl': 'Source declarations: local role mappings, pipe mapping nodes and action scalar nodes deduplicated by YAML node identity within one file. Depth/root assignment is the first reach from source-ordered local role entries. Region item counts describe each unique block declaration’s list entries; static_actions separately deduplicates aliased scalars.',
                   'B_local': 'Every local phase/role/pipe occurrence after YAML alias expansion. No cross-node inheritance.',
                   'C_effective': 'Every selected node’s phase/role/pipe after whole-area from_parent resolution through the declared direct parent. Provider and full chain retained. No runtime repeat expansion.'},
        'exact_pipe_groups': 'Stable operator:pipe:<SHA256 prefix>; SHA256 of UTF-8 compact JSON of parsed pipe with mapping and list order preserved and action strings retained. Surrounding domains, contracts and schedule are outside the fingerprint, so equal groups are not semantic equivalence. canonical_pipe is alias-expanded parsed data.',
        'group_trace': 'blocks/actions use representative source coordinates; every member carries its own block_locations, role coordinates, provider chain, role_contract_id and context_id. role_contracts[role_contract_id] contains complete info and waits_for. Follow contexts[context_id] for that member’s domain and math/schedule references; resolved_areas_ref and math_steps_ref point into files[node]. Group repeat binding details refer to the representative context only; counts resolve each member context independently.',
        'context_refs': 'All file paths relative to repository root; lines 1-based. source_node_id is node kind plus declaration character offset, scoped to one file; aliases preserve this identity. line identifies YAML declaration; occurrence_line / alias_chain identify alias sites. An inherited member’s file/path/line is in the provider. Its node is the effective consumer.',
        'repeat_forms': 'absent / domain_name / domain_slice (all bracket selections, including flattened binding g[t]) / inline_local_domain / other. Binding flags are overlapping syntactic clues, not executed trip counts or semantic proof.',
        'containers': 'pure_repeat_steady_subblocks: nested map with exactly repeat and steady. no_repeat_steady_only_blocks: any map with only steady. with_subblocks/only_subblocks distinguish containers containing child maps.',
        'list_metrics': 'Adjacent loop pairs require immediately neighboring maps with repeat. Mixed actions/loops requires an action scalar and a repeat-bearing map in the same immediate list. Region counts are immediate; loops_anywhere_under_* additionally tracks nested ancestry.',
        'status_boundary': 'semantic_eligible selects existing draft material; authored=false roots remain included. Failed skeletons and not_started entries remain manifest records and contribute no structural counts.',
        'inputs': 'Template current files are hash-compared with frozen snapshots. Extraction source is frozen snapshots. Archive docs are the semantics provenance; working schema_docs are not inputs.',
        'rerun': 'python schema_discussion/pipe_grammar/tools/census.py',
        'limits': 'Static syntax census; no GPU, kernel execution, event unrolling, runtime work estimate, semantic-equivalence claim, or design recommendation.'})
    print(f'ALL: {summary["totals"]["files"]} included files; validation={validations["passed"]}', flush=True)
    if not validations['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
