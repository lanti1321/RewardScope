"""Dependency-free, non-executing Python project discovery. Candidates are evidence, not proof."""
import ast
import os
import tomllib
from pathlib import Path
from .project_contract import inspect_project

SKIP = {'.git', '.venv', 'venv', '__pycache__', 'node_modules', 'runs', '.runtime'}


def scan_project(directory):
    root = Path(directory).resolve()
    report = {'root': str(root), 'files': 0, 'candidates': [], 'errors': [], 'scripts': {}, 'dependencies': []}
    # A declared project uses the exact contract, never heuristic fallback on failure.
    if (root / 'rl-project.json').exists():
        contract = inspect_project(root)
        report.update(contract=contract, errors=contract['errors'])
        return report
    manifest = root / 'pyproject.toml'
    if manifest.is_file():
        try:
            project = tomllib.loads(manifest.read_text()).get('project', {})
            report.update(name=project.get('name', root.name), scripts=project.get('scripts', {}), dependencies=project.get('dependencies', []))
        except (ValueError, OSError) as exc:
            report['errors'].append(str(exc))
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not Path(folder, d).is_symlink())
        for name in sorted(files):
            path = Path(folder, name)
            if path.suffix != '.py' or path.is_symlink():
                continue
            if report['files'] >= 3000:
                report['errors'].append('扫描达到 3000 文件上限'); return report
            try:
                if path.stat().st_size > 2_000_000:
                    report['errors'].append(f'{path.relative_to(root)}: 文件过大，跳过'); continue
                tree = ast.parse(path.read_text(encoding='utf-8'))
                report['files'] += 1
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Call)):
                        continue
                    label = node.name if hasattr(node, 'name') else ast.unparse(node.func)
                    lower = label.lower()
                    if isinstance(node, ast.Call):
                        leaf = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                        lower = leaf.lower()
                        if not any(token in lower for token in ('reward', 'register', 'env_cfg')) and lower not in {'linear', 'mlp', 'sequential', 'ppo', 'run_train', 'learn'}:
                            continue
                        label = leaf
                    kind = ('reward' if 'reward' in lower else 'network' if any(x in lower for x in ('actor', 'critic', 'policy', 'mlp', 'network', 'linear', 'sequential')) else 'environment' if 'env_cfg' in lower or 'register' in lower else 'training' if lower in {'train', 'learn', 'main', 'run_train'} else None)
                    if kind:
                        report['candidates'].append({'kind': kind, 'name': label, 'file': str(path.relative_to(root)), 'line': node.lineno, 'evidence': 'AST candidate'})
            except (SyntaxError, UnicodeError, OSError) as exc:
                report['errors'].append(f'{path.relative_to(root)}: {exc}')
    report['candidates'].sort(key=lambda item: (not item['file'].startswith('src/'), item['file'], item['line']))
    return report


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description='扫描 RL 项目的 Python 定义，不导入或执行项目')
    parser.add_argument('directory')
    print(json.dumps(scan_project(parser.parse_args().directory), ensure_ascii=False, indent=2))
