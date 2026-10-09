"""CLI for contract v1: init, check, train. Only train executes project code."""
import argparse
import importlib
import json
import sys
from pathlib import Path
import shutil

from .project_contract import inspect_project, resolve_config, read_json, find_project_root, validate_schema


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def verify_outputs(output, config):
    checkpoint = output / 'checkpoint.bin'
    # v1 uses a fixed opaque file name; framework format is declared by the project.
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or checkpoint.is_symlink():
        raise ValueError('training did not produce a nonempty checkpoint.bin')
    metrics = output / 'metrics.jsonl'
    previous = 0
    rows = 0
    with metrics.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if set(row) != {'step','reward_mean','components_mean'}:
                raise ValueError('invalid metric fields')
            if type(row['step']) is not int or row['step'] <= previous:
                raise ValueError('metric step must strictly increase')
            validate_schema(row['reward_mean'], {'type':'number'})
            components = row['components_mean']
            if not isinstance(components,dict) or set(components) != {r['name'] for r in config['rewards']}:
                raise ValueError('metric reward names do not match manifest')
            for value in components.values():
                validate_schema(value, {'type':'number'})
            for reward in config['rewards']:
                if reward['mode'] == 'disabled' and abs(components[reward['name']]) > 1e-8:
                    raise ValueError(f'disabled reward is not zero: {reward["name"]}')
            if abs(sum(components.values()) - row['reward_mean']) > 1e-5 * max(1, abs(row['reward_mean'])):
                raise ValueError('reward contributions do not sum to reward_mean')
            previous = row['step']
            rows += 1
    if not rows or previous < config['total_timesteps']:
        raise ValueError('training did not reach requested total_timesteps')
    return {'steps':previous, 'metric_rows':rows, 'checkpoint':'checkpoint.bin'}


def train_project(directory, task, overrides, output_dir):
    report = inspect_project(directory)
    config = resolve_config(report, task, overrides)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'config.json', config)
    write_json(output/'project.json', report)
    write_json(output/'status.json', {'status':'running'})
    try:
        # Explicit execution path. Invoke via a fresh CLI process, not in the web server.
        sys.path.insert(0, str(Path(report['root']) / report['source_root']))
        task_spec = next(t for t in report['tasks'] if t['id']==task)
        module, function = task_spec['train'].split(':')
        getattr(importlib.import_module(module), function)(config, str(output))
        result = verify_outputs(output, config)
        write_json(output/'status.json', {'status':'completed', **result})
        return result
    except BaseException as exc:
        write_json(output/'status.json', {'status':'failed', 'error':str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='copy a runnable project; never overwrite')
    init.add_argument('directory')
    check = commands.add_parser('check', help='static validation, no project code executed')
    check.add_argument('directory', nargs='?')
    train = commands.add_parser('train', help='execute trusted project code in current Python environment')
    train.add_argument('directory')
    train.add_argument('--task', required=True)
    train.add_argument('--config', help='JSON file of overrides')
    train.add_argument('--output-dir', required=True, help='new directory; must not exist')
    args = parser.parse_args()
    try:
        if args.command == 'init':
            source = Path(__file__).resolve().parents[1] / 'templates/rl_project'
            shutil.copytree(source, args.directory, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            print(f'Created {Path(args.directory).resolve()}')
        elif args.command == 'check':
            report = inspect_project(args.directory or find_project_root(Path.cwd()))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report['valid'] else 1
        else:
            result = train_project(args.directory, args.task, read_json(args.config) if args.config else {}, args.output_dir)
            print(json.dumps(result, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
