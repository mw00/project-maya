"""Linux CUDA engine + HIP expert helper setup; normal Maya configurations are untouched."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import maya_names as N

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VISIBILITY = ('HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint():
    import setup as S
    return S.source_hash((*S.ENGINE_SOURCES, 'tools/build_expert_modules.sh'))


def hip_devices(env=None):
    # Keep HIP initialization out of the server/installer's process. Runtime ordinals,
    # rather than sysfs/KFD ordering, are the ones the executor will actually use.
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-hip'],
                       env=env, capture_output=True, text=True, timeout=30)
    if p.returncode:
        raise ValueError('HIP device check failed: ' + (p.stderr.strip() or p.stdout.strip()))
    return json.loads(p.stdout)


def clean_visibility(env):
    return {k: v for k, v in env.items() if k not in VISIBILITY}


def check_budget(devices, budget):
    if len(devices) != 1:
        raise ValueError('Select exactly one visible HIP helper device.')
    free = devices[0]['free_bytes'] // 1048576
    if not 1 <= budget <= 131072 or budget > free:
        raise ValueError(f'HIP helper budget {budget} MiB exceeds available memory ({free} MiB), or is outside '
                         '1..131072. Check the UMA allocation and stop other GPU workloads; choose --helper-mib '
                         'within the actual available memory.')


def check_config(cfg):
    if not isinstance(cfg, dict) or not isinstance(cfg.get('env', {}), dict):
        raise ValueError('The input must be a Maya JSON object with an environment object.')
    env = {k: str(v) for k, v in (N.env_in(copy.deepcopy(cfg)).get('env') or {}).items()}
    args = cfg.get('args') or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ValueError('The config args must be a list of engine arguments.')
    for flag in ('--glm-pack', '--parallel'):
        if flag in args and args.index(flag) + 1 >= len(args):
            raise ValueError(f'{flag} needs a value in the input config.')
    if cfg.get('backend', 'cuda') != 'cuda' or cfg.get('family', 'glm') != 'glm' or '--glm-pack' not in args:
        raise ValueError('CUDA+HIP setup requires an existing CUDA GLM native-GGUF config (--glm-pack). '
                         'First run ./maya.sh --backend cuda --setup --no-start.')
    if '--layer-split' in args or cfg.get('layer_split') or env.get('STRATA_GLM_SPLIT', '0') not in ('', '0'):
        raise ValueError('CUDA+HIP currently supports one CUDA primary; remove layer splitting from the input config.')
    if env.get('STRATA_GLM_EXPERT_HELPER'):
        raise ValueError('Remove the native EXPERT_HELPER setting before configuring the external HIP helper.')
    ids = cfg.get('gpu', [0])
    if not isinstance(ids, list):
        ids = str(ids).split(',')
    if len(ids) != 1 or int(ids[0]) < 0:
        raise ValueError('CUDA+HIP requires exactly one CUDA primary.')
    if '--parallel' in args and args[args.index('--parallel') + 1] != '1':
        raise ValueError('This setup supports a single conversation; remove --parallel from the input config.')
    return env


def validate(cfg, env=None):
    """Fail before starting an obsolete/mismatched pair; no effect on ordinary configs.

    Passing the final child environment also checks the HIP device's free memory.
    """
    managed = cfg.get('cuda_hip')
    if not managed:
        return
    try:
        if platform.system() != 'Linux' or not isinstance(managed, dict) or managed.get('version') != 1:
            raise ValueError('This CUDA+HIP setup requires Linux and manifest version 1.')
        config_env = check_config(cfg)
        manifest = Path(managed['manifest']).resolve()
        record = json.loads(manifest.read_text())
        if record.get('version') != 1 or record.get('source') != fingerprint():
            raise ValueError('CUDA engine and HIP helper need rebuilding after this source update.')
        if cfg.get('gpu') != [record['cuda']['index']]:
            raise ValueError('The CUDA primary changed; rerun setup for the selected device.')
        expected = {'engine/maya': cfg['exe'], 'modules/libmaya_expert_hip.so':
                    config_env.get('STRATA_GLM_EXPERT_PLUGIN', '')}
        for name, actual in expected.items():
            path = manifest.parent / name
            if Path(actual).resolve() != path.resolve() or digest(path) != record['artifacts'][name]:
                raise ValueError('CUDA/HIP build is missing, changed or mismatched: ' + name)
        mask = config_env.get('HIP_VISIBLE_DEVICES', '')
        if not re.fullmatch(r'\d+', mask) or config_env.get('STRATA_GLM_EXPERT_PLUGIN_DEVICE') != '0':
            raise ValueError('The setup needs one HIP_VISIBLE_DEVICES ordinal and plugin DEVICE=0 within that mask.')
        budget = int(config_env['STRATA_GLM_EXPERT_PLUGIN_MIB'])
        share = float(config_env.get('STRATA_GLM_EXPERT_PLUGIN_PREFILL_SHARE', '0'))
        if not 1 <= budget <= 131072 or not 0 <= share <= 0.49:
            raise ValueError('Invalid helper budget (1..131072 MiB) or prefill share (0..0.49).')
        if env is not None:
            check_budget(hip_devices(env), budget)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
        raise ValueError(f'{e}\nRun ./setup-cuda-hip.sh --config <GLM-config> --hip-device <index> '
                         '--helper-mib <MiB> again. The previous pair is not started.') from e


def description(cfg):
    env = N.env_in(copy.deepcopy(cfg)).get('env') or {}
    return ('CUDA+HIP: HIP helper index ' + str(env.get('HIP_VISIBLE_DEVICES', '?')) +
            ', budget ' + str(env.get('STRATA_GLM_EXPERT_PLUGIN_MIB', '?')) +
            ' MiB; UMA shares system RAM')


def atomic_write(path, text, mode=0o600):
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def make_config(base, source_path, output, release, device, hip_device, budget, share, lib_dirs):
    env = check_config(base)
    cfg = copy.deepcopy(base)
    # Retain model/drafter choices, sampling and serving preferences; CPU tuning and
    # helper ownership from another installation do not describe this device pair.
    for k in list(env):
        if k.startswith(('STRATA_GLM_CPU_', 'STRATA_GLM_EXPERT_PLUGIN', 'STRATA_GLM_HELPER_',
                         'STRATA_GLM_RAM_', 'STRATA_GLM_PREFILL_CPU', 'STRATA_GLM_DISK_')) or \
                k in (*VISIBILITY, 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'STRATA_GLM_EXPERT_HELPER'):
            del env[k]
    env.update({'HIP_VISIBLE_DEVICES': str(hip_device), 'STRATA_GLM_SPLIT': '0',
                'STRATA_GLM_RAM_GB': '0', 'STRATA_GLM_CPU_LANE': '0', 'STRATA_GLM_PREFILL_CPU': '0',
                'STRATA_GLM_EXPERT_PLUGIN': str(release / 'modules/libmaya_expert_hip.so'),
                'STRATA_GLM_EXPERT_PLUGIN_DEVICE': '0', 'STRATA_GLM_EXPERT_PLUGIN_MIB': str(budget),
                'STRATA_GLM_EXPERT_PLUGIN_READ': '1', 'STRATA_GLM_EXPERT_PLUGIN_PREFILL_SHARE': str(share),
                'STRATA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS': '64',
                'STRATA_GLM_EXPERT_PLUGIN_PRESTAGE': '32', 'STRATA_GLM_HELPER_ZEROCOPY': '1'})
    cwd = Path(base.get('cwd') or source_path.parent).resolve()
    args = list(cfg['args'])
    for flag in ('--glm-pack', '--dflash', '--expert-profile', '--expert-profile-save'):
        if flag in args:
            i = args.index(flag) + 1
            args[i] = str((cwd / args[i]).resolve())
    for key in ('installer', 'calibration', 'speed_calibration', 'layer_split', 'cuda_hip'):
        cfg.pop(key, None)
    cfg.update(backend='cuda', family='glm', gpu=[device], exe=str(release / 'engine/maya'),
               cwd=str(ROOT), args=args, env=env, lib_dirs=lib_dirs,
               tokenizer=str((cwd / cfg['tokenizer']).resolve()), log=str(output.with_suffix('.log')),
               cuda_hip={'version': 1, 'manifest': str(release / 'pair.json')})
    return N.env_out(cfg)


def build_pair(release, gpu, nvcc, hipcc, arch, llama):
    import maya as M
    old = M.BUILD, M.EXE, M.STAMP
    try:
        M.BUILD = release / 'engine'
        M.EXE = M.BUILD / 'maya'
        M.STAMP = M.BUILD / 'MAYA-BUILD.json'
        meta = M.compile_engine([gpu['arch']], [gpu['index']], nvcc, None, llama,
                                M.S.source_hash(M.S.ENGINE_SOURCES))
    finally:
        M.BUILD, M.EXE, M.STAMP = old
    env = dict(os.environ, NVCC=nvcc, HIPCC=hipcc, HIP_ARCH=arch, CUDA_ARCH='sm_' + gpu['arch'])
    subprocess.run(['bash', str(ROOT / 'tools/build_expert_modules.sh'), str(release / 'modules')],
                   cwd=ROOT, env=env, check=True)
    return list(dict.fromkeys([*meta.get('lib_dirs', []), str(Path(hipcc).resolve().parent.parent / 'lib')]))


def interactive_config():
    """Reuse upstream model preparation, rather than a second download/pack wizard."""
    def configs():
        found = []
        for p in sorted(ROOT.glob('maya-*.json')):
            try:
                check_config(json.loads(p.read_text()))
                found.append(p)
            except (OSError, ValueError, TypeError):
                continue
        return found
    found = configs()
    if not found:
        print('No CUDA GLM config found. Starting normal Maya model setup; choose a native GGUF GLM model.')
        subprocess.run([str(ROOT / 'maya.sh'), '--backend', 'cuda', '--setup', '--plain', '--no-start',
                        '--no-speed-calibration'], cwd=ROOT, check=True)
        found = configs()
    for i, p in enumerate(found, 1):
        print(f'  {i}. {p.name}')
    answer = input('GLM configuration number or JSON path: ').strip()
    if answer.isdecimal() and 1 <= int(answer) <= len(found):
        return found[int(answer) - 1]
    if not answer:
        raise ValueError('Select a GLM configuration or provide its JSON path.')
    return Path(answer)


def toolchains(a, gpu):
    import maya as M
    nvcc, version = M.find_nvcc([int(gpu['arch'])], a.nvcc)
    lo, hi = M.nvcc_range([int(gpu['arch'])])
    if not nvcc or not version or version < lo or (hi and version >= hi):
        raise ValueError('A compatible CUDA toolkit is required: ' + M.toolkit_hint([int(gpu['arch'])]))
    nvcc = shutil.which(nvcc)
    if not nvcc:
        raise ValueError('The selected CUDA compiler is not executable.')
    if M.S.driver_major(gpu) < (580 if version >= (13, 0) else 525):
        raise ValueError('The NVIDIA driver is too old for the selected CUDA toolkit. Update the driver '
                         'and reboot before setup; --install-deps installs build tools only.')
    candidates = [a.hipcc] if a.hipcc else [
        'hipcc', str(Path(os.environ.get('ROCM_PATH', '/opt/rocm')) / 'bin/hipcc'),
        *[str(p / 'bin/hipcc') for p in sorted(Path('/opt').glob('rocm-*'), reverse=True)]]
    hipcc = next((p for c in candidates if (p := shutil.which(c))), None)
    if not hipcc or M.tool_version(hipcc) < (7, 0) or not shutil.which('g++') or M.pick_cmake() is None:
        raise ValueError('HIP 7+ hipcc, g++ and CMake 3.24+ are required. On Ubuntu 26.04, rerun '
                         './setup-cuda-hip.sh with --install-deps; otherwise use --hipcc or ROCM_PATH.')
    return nvcc, hipcc


def main(argv=None):
    ap = argparse.ArgumentParser(description='Set up an experimental CUDA primary + HIP GLM expert helper. '
                                 'Start with an existing CUDA GLM config; normal builds/configs are preserved.')
    ap.add_argument('--config', type=Path, help='existing CUDA GLM JSON config (model already set up)')
    ap.add_argument('--hip-device', type=int, help='HIP runtime ordinal from --list-devices')
    ap.add_argument('--helper-mib', type=int, help='explicit HIP allocation budget in MiB, including scratch')
    ap.add_argument('--cuda-device', type=int, help='NVIDIA index (default: the input config\'s single GPU)')
    ap.add_argument('--hip-arch', default='gfx1151', help='helper compilation target (default: tested gfx1151)')
    ap.add_argument('--prefill-share', type=float, default=0.25, help='helper row admission cap, 0..0.49 (default .25)')
    ap.add_argument('--nvcc', default=os.environ.get('NVCC'), help='CUDA compiler; otherwise use Maya discovery')
    ap.add_argument('--hipcc', default=os.environ.get('HIPCC'), help='HIP compiler; otherwise PATH/ROCM_PATH')
    ap.add_argument('--interactive', action='store_true', help='prompt for missing configuration/device/budget')
    ap.add_argument('--install-deps', action='store_true', help='Ubuntu 26.04 build packages via the shell wrapper, '
                    'then Python/CMake/Ninja in .venv; never changes GPU drivers')
    ap.add_argument('--deps-plan', action='store_true', help='show missing Ubuntu packages without installing (shell wrapper)')
    ap.add_argument('--yes', action='store_true', help='accept dependency installation; still requires device/budget choices')
    ap.add_argument('--ggml-dir', type=Path, help='existing pinned llama.cpp checkout; otherwise use Maya\'s downloader')
    ap.add_argument('--list-devices', action='store_true', help='list NVIDIA indices and HIP runtime ordinals')
    ap.add_argument('--check', action='store_true', help='validate config, tools and device memory without building')
    ap.add_argument('--no-start', action='store_true', help='build/configure only; print the launcher command')
    a = ap.parse_args(argv)
    try:
        if platform.system() != 'Linux':
            raise ValueError('CUDA+HIP setup currently supports Linux only.')
        import maya as M
        if a.deps_plan:
            raise ValueError('Use ./setup-cuda-hip.sh --deps-plan for the package preview.')
        if a.interactive and not sys.stdin.isatty():
            raise ValueError('Interactive setup needs a terminal; pass --config, --hip-device and --helper-mib instead.')
        # --check / --list-devices stay read-only. The shell installs OS packages
        # before creating .venv; reuse upstream for all Python dependencies.
        if a.install_deps and not (a.check or a.list_devices):
            if sys.prefix == sys.base_prefix:
                raise ValueError('Use ./setup-cuda-hip.sh so Python packages stay inside Maya\'s .venv.')
            M.S.pip_install(M.PY_PACKAGES, 'Maya Python and build tools')
        gpus = M.S.gpus()
        if a.list_devices:
            for g in gpus:
                print(f"CUDA {g['index']}: {g['name']}, {g['vram_gb']:.1f} GiB")
            for g in hip_devices(clean_visibility(os.environ)):
                print(f"HIP {g['index']}: {g['name']}, {g['free_bytes'] / 2**30:.1f} GiB free / "
                      f"{g['total_bytes'] / 2**30:.1f} GiB total (runtime allocation limit; UMA shares RAM)")
            return 0
        if a.interactive and not a.check:
            if a.config is None:
                a.config = interactive_config()
            if a.hip_device is None or a.helper_mib is None:
                for d in hip_devices(clean_visibility(os.environ)):
                    print(f"HIP {d['index']}: {d['name']}, {d['free_bytes'] // 1048576} MiB free (shared UMA)")
            if a.hip_device is None:
                a.hip_device = int(input('HIP helper device index: '))
            if a.helper_mib is None:
                a.helper_mib = int(input('HIP helper budget in MiB (leave room for scratch and other workloads): '))
        if a.config is None or a.hip_device is None or a.helper_mib is None:
            ap.error('--config, --hip-device and --helper-mib are required (or use --list-devices)')
        if a.hip_device < 0 or not 0 <= a.prefill_share <= 0.49 or not re.fullmatch(r'gfx[0-9a-f]+', a.hip_arch):
            raise ValueError('Invalid HIP ordinal, architecture or prefill share (expected 0..0.49).')
        source_path = a.config.expanduser().resolve()
        base = json.loads(source_path.read_text())
        check_config(base)
        ids = base.get('gpu', [0])
        if not isinstance(ids, list):
            ids = [int(i) for i in str(ids).split(',')]
        if len(ids) != 1:
            raise ValueError('Input config must select exactly one CUDA primary.')
        device = a.cuda_device if a.cuda_device is not None else int(ids[0])
        gpu = next((g for g in gpus if g['index'] == device), None)
        if gpu is None:
            raise ValueError(f'CUDA device {device} was not found; use --list-devices.')
        cwd = Path(base.get('cwd') or source_path.parent).resolve()
        args = base['args']; pack = (cwd / args[args.index('--glm-pack') + 1]).resolve()
        if not all((pack / f).is_file() for f in ('index.txt', 'native_experts.txt', 'dense.bin')):
            raise ValueError('A native GGUF GLM pack is required; EXL3 and unpacked GGUF files are unsupported.')
        if not (cwd / base.get('tokenizer', '') / 'vocab.json').is_file():
            raise ValueError('The input config\'s tokenizer/vocab.json is missing.')
        nvcc, hipcc = toolchains(a, gpu)
        env = dict(clean_visibility(os.environ), HIP_VISIBLE_DEVICES=str(a.hip_device))
        devices = hip_devices(env); check_budget(devices, a.helper_mib)
        print(f"CUDA primary: {gpu['name']} (index {device}); HIP helper: {devices[0]['name']} "
              f"(runtime index {a.hip_device}), {a.helper_mib} MiB budget.")
        if a.check:
            print('Preflight passed; no files or builds changed.')
            return 0
        name = source_path.stem
        if not name.startswith('maya-'):
            name = 'maya-' + name
        if not name.endswith('-cuda-hip'):
            name += '-cuda-hip'
        output = ROOT / (name + '.json')
        launcher = ROOT / ('run-' + name + '.sh')
        if output.exists() and not json.loads(output.read_text()).get('cuda_hip'):
            raise ValueError(f'Refusing to overwrite an ordinary configuration: {output}')
        if launcher.exists() and '# Maya CUDA+HIP launcher' not in launcher.read_text():
            raise ValueError(f'Refusing to overwrite an unrelated launcher: {launcher}')
        llama = a.ggml_dir.expanduser().resolve() if a.ggml_dir else M.S.get_llama_cpp()
        if not (llama / 'ggml/CMakeLists.txt').is_file():
            raise ValueError('The pinned llama.cpp checkout needs ggml/CMakeLists.txt.')
        builds = ROOT / 'build-cuda-hip'; builds.mkdir(exist_ok=True)
        release = Path(tempfile.mkdtemp(prefix='pair-', dir=builds))
        source = fingerprint()
        print('Building a separate pair in', release)
        lib_dirs = build_pair(release, gpu, nvcc, hipcc, a.hip_arch, llama)
        if source != fingerprint():
            raise ValueError('Source changed during the build; rerun setup. Existing configuration was preserved.')
        artifacts = {f: digest(release / f) for f in ('engine/maya', 'modules/libmaya_expert_hip.so',
                                                      'modules/libmaya_expert_cuda.so')}
        atomic_write(release / 'pair.json', json.dumps({'version': 1, 'source': source, 'artifacts': artifacts,
                     'cuda': gpu, 'hip': devices[0], 'hip_arch': a.hip_arch}, indent=2) + '\n')
        cfg = make_config(base, source_path, output, release, device, a.hip_device, a.helper_mib,
                          a.prefill_share, lib_dirs)
        validate(cfg)
        script = '#!/bin/sh\n# Maya CUDA+HIP launcher\nexec ' + shlex.join([str(ROOT / 'maya.sh'), '--config',
                                                                          str(output)]) + ' "$@"\n'
        atomic_write(launcher, script, 0o755)
        atomic_write(output, json.dumps(cfg, indent=2) + '\n')
        print('CUDA+HIP configuration:', output)
        print('Start:', shlex.quote(str(launcher)))
        print('Benchmark:', shlex.join([str(launcher), '--bench']))
        print('After source updates, rerun setup to rebuild both components together.')
        return 0 if a.no_start else subprocess.call([str(launcher)])
    except (OSError, ValueError, KeyError, IndexError, TypeError, EOFError, subprocess.SubprocessError) as e:
        print('CUDA+HIP setup: ' + str(e), file=sys.stderr)
        return 1


def probe():
    import ctypes as C
    import ctypes.util
    hip = C.CDLL(ctypes.util.find_library('amdhip64') or 'libamdhip64.so')
    def ck(rc):
        if rc:
            raise RuntimeError(f'HIP runtime error {rc}; check the driver and device visibility.')
    ck(hip.hipInit(0)); count = C.c_int(); ck(hip.hipGetDeviceCount(C.byref(count)))
    found = []
    for i in range(count.value):
        ck(hip.hipSetDevice(i)); name = C.create_string_buffer(256)
        ck(hip.hipDeviceGetName(name, len(name), i))
        free, total = C.c_size_t(), C.c_size_t()
        ck(hip.hipMemGetInfo(C.byref(free), C.byref(total)))
        found.append({'index': i, 'name': name.value.decode(), 'free_bytes': free.value, 'total_bytes': total.value})
    print(json.dumps(found))


if __name__ == '__main__':
    if sys.argv[1:] == ['--probe-hip']:
        try:
            probe()
        except Exception as e:
            print(str(e), file=sys.stderr); sys.exit(1)
    else:
        sys.exit(main())
