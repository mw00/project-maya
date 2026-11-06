"""Paired installer/launcher regressions: no GPU, compiler, network or model download required."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya
import maya_cuda_hip as H
from serve import models, server


class CudaHipSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='maya pair ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pack = self.root / 'model pack'; self.pack.mkdir()
        for name in ('index.txt', 'native_experts.txt', 'dense.bin'):
            (self.pack / name).touch()
        (self.pack / 'tokenizer').mkdir(); (self.pack / 'tokenizer/vocab.json').write_text('{}')
        self.llama = self.root / 'llama'; (self.llama / 'ggml').mkdir(parents=True)
        (self.llama / 'ggml/CMakeLists.txt').touch()
        self.base = {'backend': 'cuda', 'family': 'glm', 'gpu': [2], 'cwd': str(self.root),
                     'exe': str(self.root / 'build/maya'), 'tokenizer': 'model pack/tokenizer',
                     'args': ['--glm-pack', 'model pack', '--max-context', '9216', '--dflash', 'drafter.gguf',
                              '--expert-profile', 'profile.bin'],
                     'env': {'MAYA_GLM_NO_MTP': '1', 'MAYA_GLM_CPU_LANE': '16', 'ROCR_VISIBLE_DEVICES': '7'},
                     'sampling': {'temp': 0.7}, 'port': 9000}
        self.input = self.root / 'maya-glm.json'; self.input.write_text(json.dumps(self.base))
        self.output = self.root / 'maya-glm-cuda-hip.json'
        self.gpu = {'index': 2, 'arch': '120', 'name': 'RTX test', 'driver': '595.0'}
        self.devices = [{'index': 0, 'name': 'HIP test', 'free_bytes': 4 << 30, 'total_bytes': 4 << 30}]
        self.builds = []
        for p in [patch.object(H, 'ROOT', self.root), patch.object(H.platform, 'system', return_value='Linux'),
                  patch.object(H, 'fingerprint', return_value='source-a'),
                  patch.object(H, 'hip_devices', return_value=self.devices),
                  patch.object(maya.S, 'gpus', return_value=[self.gpu]),
                  patch.object(maya, 'pick_cmake', return_value='/usr/bin/cmake'),
                  patch.object(maya, 'find_nvcc', return_value=('/cuda/bin/nvcc', (13, 1))),
                  patch.object(maya, 'tool_version', return_value=(7, 1)),
                  patch.object(H.shutil, 'which', side_effect=lambda x: '/tools/' + Path(x).name),
                  patch.object(H, 'build_pair', side_effect=self.fake_build)]:
            p.start(); self.addCleanup(p.stop)
        self.args = ['--config', str(self.input), '--hip-device', '1', '--helper-mib', '1024',
                     '--ggml-dir', str(self.llama), '--no-start']

    def fake_build(self, release, *args):
        self.builds.append(release)
        (release / 'engine').mkdir(); (release / 'modules').mkdir()
        for f in ('engine/maya', 'modules/libmaya_expert_hip.so', 'modules/libmaya_expert_cuda.so'):
            (release / f).write_text(str(release) + f)
        return ['/cuda/lib', '/hip/lib']

    def install(self):
        self.assertEqual(H.main(self.args), 0)
        return json.loads(self.output.read_text())

    def test_new_generations_do_not_overwrite_either_normal_backend_or_old_pair(self):
        originals = {self.input: self.input.read_bytes()}
        for name in ('build/maya', 'build-hip/maya', 'build/MAYA-BUILD.json', 'build-hip/MAYA-BUILD.json'):
            path = self.root / name; path.parent.mkdir(exist_ok=True); path.write_text('ordinary ' + name)
            originals[path] = path.read_bytes()
        first = self.install(); first_pair = Path(first['cuda_hip']['manifest']).parent
        original_pair = {p: p.read_bytes() for p in first_pair.rglob('*') if p.is_file()}
        second = self.install()
        self.assertNotEqual(first['exe'], second['exe'])
        for path, data in {**originals, **original_pair}.items():
            self.assertEqual(path.read_bytes(), data)
        launcher = self.root / 'run-maya-glm-cuda-hip.sh'
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertIn('"$@"', launcher.read_text())
        self.assertIn("'" + str(self.output) + "'", launcher.read_text())
        subprocess.run(['sh', '-n', str(launcher)], check=True)

    def test_failed_helper_build_keeps_previous_config_launcher_and_pair(self):
        old = self.install()
        before = {p: p.read_bytes() for p in (self.output, self.root / 'run-maya-glm-cuda-hip.sh', Path(old['exe']))}
        with patch.object(H, 'build_pair', side_effect=subprocess.CalledProcessError(1, 'hipcc')):
            self.assertEqual(H.main(self.args), 1)
        for p, contents in before.items():
            self.assertEqual(p.read_bytes(), contents)

    def test_preserves_model_and_drafter_choices_resolves_paths_and_separates_ordinals(self):
        cfg = self.install(); env = maya.N.env_in(copy.deepcopy(cfg))['env']
        self.assertEqual(cfg['gpu'], [2]); self.assertEqual(env['HIP_VISIBLE_DEVICES'], '1')
        self.assertEqual(env['STRATA_GLM_EXPERT_PLUGIN_DEVICE'], '0')
        self.assertEqual(env['STRATA_GLM_NO_MTP'], '1')
        self.assertEqual(env['STRATA_GLM_CPU_LANE'], '0'); self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
        self.assertEqual(cfg['port'], 9000); self.assertEqual(cfg['sampling'], self.base['sampling'])
        self.assertEqual(cfg['tokenizer'], str(self.pack / 'tokenizer'))
        self.assertIn(str(self.root / 'drafter.gguf'), cfg['args'])
        self.assertTrue(all(not k.startswith('STRATA_') for k in cfg['env']))
        self.assertEqual(json.loads(self.input.read_text()), self.base)

    def test_check_does_not_build_download_or_write(self):
        with patch.object(maya.S, 'get_llama_cpp', side_effect=AssertionError('no download')), \
                patch.object(maya.S, 'pip_install', side_effect=AssertionError('no installation')):
            self.assertEqual(H.main([*self.args, '--install-deps', '--check']), 0)
        self.assertFalse(self.output.exists()); self.assertEqual(self.builds, [])
        self.assertFalse((self.root / 'build-cuda-hip').exists())

    def test_dependency_install_reuses_upstream_python_environment_before_build(self):
        events = []
        def build(*args):
            events.append('build'); return self.fake_build(*args)
        with patch.object(H.sys, 'prefix', '/private/venv'), \
                patch.object(maya.S, 'pip_install', side_effect=lambda *a: events.append('pip')) as pip, \
                patch.object(H, 'build_pair', side_effect=build):
            self.assertEqual(H.main([*self.args, '--install-deps']), 0)
        self.assertEqual(events, ['pip', 'build'])
        self.assertEqual(pip.call_args.args[0], maya.PY_PACKAGES)

    def test_old_cuda_and_hip_toolchains_fail_before_build(self):
        with patch.object(maya, 'find_nvcc', return_value=('/cuda/bin/nvcc', (12, 4))):
            self.assertEqual(H.main(self.args), 1)
        with patch.object(maya, 'tool_version', return_value=(6, 4)):
            self.assertEqual(H.main(self.args), 1)
        with patch.object(maya.S, 'driver_major', return_value=550):
            self.assertEqual(H.main(self.args), 1)
        self.assertEqual(self.builds, [])

    def test_interactive_flow_selects_config_device_and_budget_then_builds(self):
        args = ['--interactive', '--ggml-dir', str(self.llama), '--no-start']
        with patch.object(H.sys.stdin, 'isatty', return_value=True), \
                patch('builtins.input', side_effect=['1', '1', '1024']):
            self.assertEqual(H.main(args), 0)
        cfg = json.loads(self.output.read_text())
        self.assertEqual(cfg['env']['HIP_VISIBLE_DEVICES'], '1')
        self.assertEqual(cfg['env']['MAYA_GLM_EXPERT_PLUGIN_MIB'], '1024')

    def test_first_model_reuses_stock_setup_and_preserves_its_config(self):
        self.input.unlink()
        def setup(cmd, **kwargs):
            self.assertIn('--backend', cmd); self.assertIn('cuda', cmd)
            self.assertIn('--no-start', cmd)
            self.input.write_text(json.dumps(self.base))
        with patch.object(H.subprocess, 'run', side_effect=setup), patch('builtins.input', return_value='1'):
            self.assertEqual(H.interactive_config(), self.input)
        self.assertEqual(json.loads(self.input.read_text()), self.base)

    def test_interactive_without_terminal_never_installs_or_builds(self):
        with patch.object(H.sys.stdin, 'isatty', return_value=False), \
                patch.object(maya.S, 'pip_install', side_effect=AssertionError('no installation')):
            self.assertEqual(H.main(['--interactive', '--install-deps']), 1)
        self.assertEqual(self.builds, [])

    def test_oversized_budget_fails_before_build(self):
        args = list(self.args); args[args.index('--helper-mib') + 1] = '90112'
        self.assertEqual(H.main(args), 1); self.assertEqual(self.builds, [])

    def test_refuses_unrelated_configuration_and_launcher(self):
        self.output.write_text('{}')
        self.assertEqual(H.main(self.args), 1); self.assertEqual(self.output.read_text(), '{}')
        self.output.unlink()
        launcher = self.root / 'run-maya-glm-cuda-hip.sh'; launcher.write_text('# unrelated')
        self.assertEqual(H.main(self.args), 1); self.assertEqual(launcher.read_text(), '# unrelated')
        self.assertEqual(self.builds, [])

    def test_rejects_non_glm_non_cuda_split_and_exl3_inputs(self):
        for change in ({'family': 'deepseek'}, {'backend': 'hip'}, {'gpu': [0, 2]}, {'layer_split': 'auto'},
                       {'env': {'MAYA_GLM_EXPERT_HELPER': '1'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                H.check_config({**self.base, **change})
        (self.pack / 'native_experts.txt').unlink()
        self.assertEqual(H.main(self.args), 1); self.assertEqual(self.builds, [])

    def test_rejects_invalid_share_and_arch(self):
        for options in (['--prefill-share', 'nan'], ['--prefill-share', '.5'], ['--hip-arch', 'bad;arch']):
            with self.subTest(options=options):
                self.assertEqual(H.main([*self.args, *options]), 1)
        self.assertEqual(self.builds, [])

    def test_source_change_during_build_never_publishes_config(self):
        with patch.object(H, 'fingerprint', side_effect=['before', 'after']):
            self.assertEqual(H.main(self.args), 1)
        self.assertFalse(self.output.exists())

    def test_start_guard_rejects_stale_source_mismatched_binary_and_missing_module(self):
        cfg = self.install(); H.validate(cfg)
        with patch.object(H, 'fingerprint', return_value='updated'), self.assertRaisesRegex(ValueError, 'rebuilding'):
            H.validate(cfg)
        binary = Path(cfg['exe']); old = binary.read_bytes(); binary.write_text('other backend')
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            H.validate(cfg)
        binary.write_bytes(old)
        Path(cfg['env']['MAYA_GLM_EXPERT_PLUGIN']).unlink()
        with self.assertRaises(ValueError):
            H.validate(cfg)

    def test_cannot_redirect_generated_config_to_normal_backend_or_other_cuda_gpu(self):
        cfg = self.install()
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            H.validate({**cfg, 'exe': str(self.root / 'build/maya')})
        with self.assertRaisesRegex(ValueError, 'primary changed'):
            H.validate({**cfg, 'gpu': [0]})

    def test_runtime_guard_clears_foreign_visibility_and_checks_current_memory(self):
        cfg = self.install()
        with patch.dict(os.environ, {'ROCR_VISIBLE_DEVICES': '7', 'HIP_VISIBLE_DEVICES': '5'}), \
                patch.object(H, 'hip_devices', return_value=self.devices) as probe:
            env = server.child_env(cfg)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
            self.assertEqual(env['HIP_VISIBLE_DEVICES'], '1'); self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
            self.assertEqual(probe.call_args.args[0]['HIP_VISIBLE_DEVICES'], '1')
        with patch.object(H, 'hip_devices', return_value=[{**self.devices[0], 'free_bytes': 256 << 20}]), \
                self.assertRaisesRegex(ValueError, 'budget'):
            server.child_env(cfg)

    def test_plain_configs_do_not_probe_hip_or_change_their_visibility(self):
        with patch.object(H, 'hip_devices', side_effect=AssertionError('plain CUDA must not initialize HIP')), \
                patch.dict(os.environ, {'HIP_VISIBLE_DEVICES': '7'}):
            self.assertEqual(server.child_env(self.base)['HIP_VISIBLE_DEVICES'], '7')
            H.validate(self.base)

    def test_managed_refresh_never_builds_only_cuda_and_skips_automatic_calibration(self):
        cfg = self.install()
        with patch.object(maya, 'compile_engine', side_effect=AssertionError('must rebuild the pair')):
            maya.refresh_engine(cfg)
        self.assertFalse(maya.needs_speed_calibration(cfg))
        self.assertFalse(maya.calibrate_config(self.output))
        self.assertIn('CUDA + HIP', models.quant_label(cfg, self.output))
        with self.assertRaisesRegex(ValueError, 'setup-cuda-hip'):
            models.apply(cfg, {'gpus': [0]})

    def test_malformed_configs_fail_with_actionable_errors(self):
        for cfg in ([], {**self.base, 'env': []}, {**self.base, 'args': '--glm-pack pack'},
                    {**self.base, 'args': ['--glm-pack']}, {**self.base, 'args': ['--glm-pack', 'p', '--parallel']}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                H.check_config(cfg)
        H.check_config({**self.base, 'env': {'MAYA_GLM_SPLIT': 0}})

    def test_ui_rejects_invalid_helper_settings_before_saving(self):
        cfg = self.install()
        before = self.output.read_bytes()
        for change in ({'MAYA_GLM_EXPERT_PLUGIN_PREFILL_SHARE': '0.9'},
                       {'MAYA_GLM_EXPERT_PLUGIN_MIB': '0'},
                       {'MAYA_GLM_EXPERT_PLUGIN': '/other/module.so'}):
            env = {**cfg['env'], **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                models.save(self.output, {'env': env})
            self.assertEqual(self.output.read_bytes(), before)

    def test_reports_helper_without_double_counting_uma(self):
        cfg = self.install()
        self.assertIn('HIP helper index 1', H.description(cfg))
        self.assertIn('UMA shares system RAM', H.description(cfg))


class BuildPairTests(unittest.TestCase):
    def test_actual_build_orchestrator_isolates_engine_stamp_and_module_output(self):
        with tempfile.TemporaryDirectory() as d:
            release = Path(d) / 'new pair'
            before = maya.BUILD, maya.EXE, maya.STAMP
            def compile_engine(*args):
                self.assertEqual(maya.BUILD, release / 'engine')
                self.assertEqual(maya.EXE, release / 'engine/maya')
                self.assertEqual(maya.STAMP, release / 'engine/MAYA-BUILD.json')
                return {'lib_dirs': ['/cuda/lib']}
            with patch.object(maya, 'compile_engine', side_effect=compile_engine), \
                    patch.object(maya.S, 'source_hash', return_value='source'), patch.object(H.subprocess, 'run') as run:
                H.build_pair(release, {'index': 2, 'arch': '120'}, '/cuda/bin/nvcc', '/rocm/bin/hipcc',
                             'gfx1151', Path('/llama'))
            self.assertEqual(run.call_args.args[0][-1], str(release / 'modules'))
            self.assertEqual(run.call_args.kwargs['env']['CUDA_ARCH'], 'sm_120')
            self.assertEqual(run.call_args.kwargs['env']['HIP_ARCH'], 'gfx1151')
            self.assertEqual((maya.BUILD, maya.EXE, maya.STAMP), before)


class DependencyShellTests(unittest.TestCase):
    """Exercise the real wrapper against a fake package manager; never run sudo."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='maya deps ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'os-release').write_text('ID=ubuntu\nVERSION_ID=26.04\n')
        script = (H.ROOT / 'setup-cuda-hip.sh').read_text().replace('/etc/os-release', shlex.quote(str(self.root / 'os-release')))
        (self.root / 'setup-cuda-hip.sh').write_text(script)
        self.log = self.root / 'calls'
        self.bin = self.root / 'bin'; self.bin.mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        CALL_LOG=str(self.log), MISSING='hipcc')
        self.command('uname', 'echo x86_64')
        self.command('dpkg-query', '[ "$3" = "$MISSING" ] || printf "install ok installed"')
        self.command('apt-get', 'echo "apt-get $*" >> "$CALL_LOG"; echo "${PLAN_OUTPUT:-Inst hipcc (7.1 Ubuntu)}"')
        self.command('sudo', 'echo "sudo $*" >> "$CALL_LOG"; "$@"')
        (self.root / 'maya.sh').write_text('#!/bin/sh\necho "maya $*" >> "$CALL_LOG"\n')
        (self.root / 'maya.sh').chmod(0o755)

    def command(self, name, body):
        p = self.bin / name; p.write_text('#!/bin/sh\n' + body + '\n'); p.chmod(0o755)

    def run_setup(self, *args, **env):
        p = subprocess.run(['bash', str(self.root / 'setup-cuda-hip.sh'), *args],
                           env={**self.env, **env}, input='', capture_output=True, text=True)
        return p, self.log.read_text().splitlines() if self.log.exists() else []

    def test_plan_never_runs_sudo_installs_or_starts_python(self):
        p, calls = self.run_setup('--deps-plan')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(calls, ['apt-get --simulate --no-install-recommends install hipcc'])

    def test_install_orders_update_simulation_install_then_launcher(self):
        p, calls = self.run_setup('--install-deps', '--yes')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(calls, ['sudo apt-get update', 'apt-get update',
                                'apt-get --simulate --no-install-recommends install hipcc',
                                'sudo apt-get install -y --no-remove --no-install-recommends hipcc',
                                'apt-get install -y --no-remove --no-install-recommends hipcc',
                                'maya --setup-cuda-hip --install-deps --yes'])

    def test_installed_packages_are_not_reinstalled(self):
        p, calls = self.run_setup('--install-deps', '--yes', MISSING='none')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(calls, ['maya --setup-cuda-hip --install-deps --yes'])

    def test_driver_changes_or_package_removal_stop_before_install(self):
        for plan in ('Remv old-package', 'Inst nvidia-driver-590 (590 Ubuntu)', 'Inst amdgpu-dkms (7 Ubuntu)'):
            if self.log.exists(): self.log.unlink()
            p, calls = self.run_setup('--install-deps', '--yes', PLAN_OUTPUT=plan)
            self.assertNotEqual(p.returncode, 0)
            self.assertFalse(any('apt-get install ' in c or c.startswith('maya ') for c in calls))

    def test_readonly_options_do_not_install_even_with_install_flag(self):
        for option in ('--check', '--list-devices', '--help'):
            if self.log.exists(): self.log.unlink()
            p, calls = self.run_setup('--install-deps', option)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(calls, [f'maya --setup-cuda-hip --install-deps {option}'])

    def test_no_terminal_or_declined_install_never_changes_packages(self):
        for args in ([], ['--install-deps']):
            p, calls = self.run_setup(*args)
            self.assertNotEqual(p.returncode, 0)
            self.assertEqual(calls, [])

    def test_unsupported_os_never_changes_packages(self):
        (self.root / 'os-release').write_text('ID=ubuntu\nVERSION_ID=24.04\n')
        p, calls = self.run_setup('--install-deps', '--yes')
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
