"""Versioned static project contract. Never imports or executes project code."""
import ast
import json
import math
import re
import tomllib
from pathlib import Path, PurePosixPath

SCHEMA_PATH = Path(__file__).resolve().parents[1] / 'schemas/rl-project-v1.schema.json'


def find_project_root(directory):
    root = Path(directory).resolve()
    for candidate in (root, *root.parents):
        if (candidate / 'rl-project.json').is_file():
            return candidate
    return root


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    path = Path(path)
    if path.stat().st_size > 1_000_000:
        raise ValueError(f'JSON exceeds 1 MB: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'invalid number: {x}')))


def validate_schema(value, schema, location='$'):
    """Evaluate the keywords used in our bundled schema, without dependencies."""
    kind = schema['type']
    valid = {'object': isinstance(value, dict), 'array': isinstance(value, list),
             'string': isinstance(value, str), 'boolean': type(value) is bool,
             'integer': type(value) is int,
             'number': type(value) in (int, float) and math.isfinite(value)}[kind]
    if not valid:
        raise ValueError(f'{location}: expected {kind}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{location}: allowed values {schema["enum"]}')
    if kind == 'object':
        missing = set(schema['required']) - value.keys()
        extra = value.keys() - schema['properties'].keys()
        if missing or extra:
            raise ValueError(f'{location}: missing={sorted(missing)}, unknown={sorted(extra)}')
        for key, child in value.items():
            validate_schema(child, schema['properties'][key], f'{location}.{key}')
    elif kind == 'array':
        if len(value) < schema.get('minItems', 0):
            raise ValueError(f'{location}: too few items')
        if schema.get('uniqueItems') and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise ValueError(f'{location}: duplicate items')
        for index, child in enumerate(value):
            validate_schema(child, schema['items'], f'{location}[{index}]')
    elif kind == 'string':
        if len(value.strip()) < schema.get('minLength', 0) or ('pattern' in schema and not re.fullmatch(schema['pattern'], value)):
            raise ValueError(f'{location}: invalid string')
    elif kind in ('integer', 'number'):
        if ('minimum' in schema and value < schema['minimum']) or ('exclusiveMinimum' in schema and value <= schema['exclusiveMinimum']):
            raise ValueError(f'{location}: below minimum')


def inside(root, relative):
    if not isinstance(relative, str) or '\\' in relative or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
        raise ValueError(f'invalid project relative path: {relative}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'path escapes project: {relative}')
    return path


def function_source(root, source, reference, arguments):
    module, name = reference.split(':')
    file = inside(root, f'{source}/{module.replace(".", "/")}.py')
    if not file.is_file():
        raise ValueError(f'{reference}: source file missing')
    if file.stat().st_size > 2_000_000:
        raise ValueError(f'{reference}: source too large')
    tree = ast.parse(file.read_text(encoding='utf-8'))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(definitions) != 1:
        raise ValueError(f'{reference}: must define exactly one top-level synchronous function')
    node = definitions[0]
    actual = [arg.arg for arg in node.args.args]
    if actual != arguments or node.args.posonlyargs or node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.decorator_list:
        raise ValueError(f'{reference}: expected undecorated signature ({", ".join(arguments)})')
    return {'file': str(file.relative_to(root)), 'line': node.lineno}


def inspect_project(directory):
    root = Path(directory).resolve()
    report = {'root': str(root), 'manifest': 'rl-project.json', 'valid': False,
              'validation': 'static', 'runtime_verified': False, 'errors': [], 'tasks': []}
    try:
        package_file = inside(root, 'pyproject.toml')
        if package_file.stat().st_size > 1_000_000:
            raise ValueError('pyproject.toml exceeds 1 MB')
        package = tomllib.loads(package_file.read_text(encoding='utf-8')).get('project', {})
        if not isinstance(package, dict) or not all(isinstance(package.get(key), str) and package[key].strip() for key in ('name', 'requires-python')):
            raise ValueError('pyproject.toml requires project.name and project.requires-python')
        if not isinstance(package.get('dependencies'), list) or not all(isinstance(item, str) and item.strip() for item in package['dependencies']):
            raise ValueError('pyproject.toml requires project.dependencies array')
        if not inside(root, 'README.md').is_file():
            raise ValueError('README.md missing')
        manifest_file = inside(root, 'rl-project.json')
        manifest = read_json(manifest_file)
        validate_schema(manifest, read_json(SCHEMA_PATH))
        source = manifest['source_root']
        if not inside(root, source).is_dir():
            raise ValueError('source_root does not exist')
        ids = set()
        for task in manifest['tasks']:
            if task['id'] in ids:
                raise ValueError(f'duplicate task id: {task["id"]}')
            ids.add(task['id'])
            names = [reward['name'] for reward in task['rewards']]
            if len(set(names)) != len(names):
                raise ValueError(f'{task["id"]}: duplicate reward names')
            task['sources'] = {
                'environment': function_source(root, source, task['environment'], ['config']),
                'train': function_source(root, source, task['train'], ['config', 'output_dir']),
                'rewards': {r['name']: function_source(root, source, r['function'], ['transition']) for r in task['rewards']},
            }
        report.update(valid=True, project=manifest['project'], schema_version=1,
                      source_root=source, tasks=manifest['tasks'])
    except (ValueError, OSError, SyntaxError, RecursionError, OverflowError) as exc:
        report['errors'].append(str(exc))
    return report


def resolve_config(report, task_id, overrides=None):
    if not report['valid']:
        raise ValueError('project contract is invalid')
    task = next((t for t in report['tasks'] if t['id'] == task_id), None)
    if task is None:
        raise ValueError('unknown task id')
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict) or set(overrides) - {'num_envs','total_timesteps','seed','network','reward_weights','disabled_rewards'}:
        raise ValueError('unknown configuration overrides')
    defaults = json.loads(json.dumps(task['defaults']))
    defaults.update({k:v for k,v in overrides.items() if k in defaults})
    schema = read_json(SCHEMA_PATH)['properties']['tasks']['items']['properties']['defaults']
    validate_schema(defaults, schema, '$.config')
    weights = overrides.get('reward_weights', {})
    disabled = overrides.get('disabled_rewards', [])
    names = {r['name'] for r in task['rewards']}
    if not isinstance(weights, dict) or set(weights) - names:
        raise ValueError('reward_weights must contain known reward names')
    for value in weights.values():
        validate_schema(value, {'type':'number'}, '$.reward_weights')
    if not isinstance(disabled, list) or any(not isinstance(x,str) for x in disabled) or len(set(disabled)) != len(disabled) or set(disabled) - names:
        raise ValueError('disabled_rewards must contain unique known reward names')
    rewards = []
    for reward in task['rewards']:
        item = dict(reward)
        item['mode'] = 'disabled' if reward['name'] in disabled or not reward['enabled'] else 'fixed' if reward['name'] in weights else 'default'
        item['weight'] = 0.0 if item['mode']=='disabled' else weights.get(reward['name'], reward['weight'])
        rewards.append(item)
    return {'schema_version':1, 'task':task_id, **defaults, 'step_dt':task['step_dt'],
            'reward_scale':task['reward_scale'], 'rewards':rewards}
