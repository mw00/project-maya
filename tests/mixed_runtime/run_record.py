#!/usr/bin/env python3
"""Run PB1 probe with source verification and retain runtime-library diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-source', type=Path, required=True)
    parser.add_argument('--cuda-device', type=int, required=True)
    parser.add_argument('--hip-device', type=int, required=True)
    args = parser.parse_args()
    src = Path(__file__).resolve().parent
    expected = json.loads(args.expected_source.read_text())
    for name, checksum in expected.items():
        if Path(name).name != name or digest(src / name) != checksum:
            raise RuntimeError('source identity mismatch: ' + name)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.build.resolve() / 'maya_mixed_probe'
    plugin = args.build.resolve() / 'libmaya_hip_probe.so'
    manifest = {'sources': expected, 'binaries': {str(p): digest(p) for p in (binary, plugin)}}
    (out / 'identity.json').write_text(json.dumps(manifest, indent=2) + '\n')
    commands = [
        ['uname', '-a'], ['free', '-b'],
        ['nvidia-smi', '--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,memory.used', '--format=csv'],
        ['rocminfo'], ['hipcc', '--version'], ['/usr/local/cuda/bin/nvcc', '--version'],
        ['ldd', str(binary)], ['ldd', str(plugin)], ['nm', '-D', '--defined-only', str(plugin)],
    ]
    for i, command in enumerate(commands):
        with (out / f'environment-{i}.log').open('w') as log:
            log.write(json.dumps(command) + '\n'); log.flush()
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                log.write(f'\nexit_code={result.returncode}\n')
            except (OSError, subprocess.TimeoutExpired) as error:
                log.write(str(error) + '\n')
    command = [str(binary), str(plugin), str(args.cuda_device), str(args.hip_device)]
    (out / 'command.json').write_text(json.dumps(command) + '\n')
    env = dict(os.environ, LD_DEBUG='libs,files')
    with (out / 'run.log').open('w') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=120)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            log.write('\nprobe timeout\n')
    # Linker diagnostics show actual loaded libraries, not merely headers/ldd.
    libraries = {}
    for line in (out / 'run.log').read_text(errors='replace').splitlines():
        if 'calling init:' in line:
            path = Path(line.split('calling init:', 1)[1].strip())
            if path.is_file():
                libraries[str(path.resolve())] = digest(path)
    (out / 'runtime-libraries.json').write_text(json.dumps(libraries, indent=2) + '\n')
    (out / 'exit-code.txt').write_text(str(rc) + '\n')
    return rc


if __name__ == '__main__':
    sys.exit(main())
