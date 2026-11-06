#!/usr/bin/env python3
"""Diagnostic real-GGUF CUDA/HIP expert replay. Cross-backend errors are measurements,
not a preapproved quality tolerance. This does not run an inference engine."""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import numpy as np
from gguf import GGUFReader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cuda-device', type=int, required=True)
    ap.add_argument('--hip-device', type=int, required=True)
    ap.add_argument('--persistent', action='store_true')
    ap.add_argument('--adapter', type=Path)
    # Adapter mode exercises the host loader with the previously validated executor DSOs.
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inp = json.loads(args.inputs.read_text())
    layer, width, k = inp['layer'], inp['width'], inp['top_k']
    assert width == 4096 and k == 8
    x = np.asarray(inp['input'], dtype=np.float32).reshape(-1, width)
    ids = np.asarray(inp['selected'], dtype=np.int32).reshape(-1, k)
    routes = np.asarray(inp['routing_weights'], dtype=np.float32).reshape(-1, k)
    names = [f'blk.{layer}.ffn_{role}_exps.weight' for role in ['gate', 'up', 'down']]
    tensors, readers, source_info = {}, [], {}
    for path in sorted(args.model_dir.glob('*.gguf')):
        reader = GGUFReader(str(path), 'r')
        readers.append(reader)  # keep mappings alive until extraction finishes
        for t in reader.tensors:
            if t.name in names:
                tensors[t.name] = t
                source_info[t.name] = {'file': str(path), 'file_size': path.stat().st_size,
                    'shape': t.shape.tolist(), 'type': int(t.tensor_type), 'offset': int(t.data_offset)}
    assert set(tensors) == set(names)
    gate, up, down = [tensors[n] for n in names]
    assert gate.tensor_type == up.tensor_type
    assert list(gate.shape) == list(up.shape) == [4096, 2048, inp['experts']]
    assert list(down.shape) == [2048, 4096, inp['experts']]
    raw = [t.data.view(np.uint8).reshape(inp['experts'], -1) for t in (gate, up, down)]
    offset = raw[0].shape[1] + raw[1].shape[1]
    stride = offset + raw[2].shape[1]
    manifest = {'inputs_sha256': sha(args.inputs), 'weights': source_info,
        'libraries': {}, 'fixture_blobs': {}, 'clamp': 8.0,
        'scope': 'isolated decode expert kernels, four captured token inputs; not full model quality'}
    assert not (args.adapter and args.persistent)
    functions = {}
    if args.adapter:
        manifest['adapter'] = {'path':str(args.adapter.resolve()),'sha256':sha(args.adapter)}
    for backend in ['cuda', 'hip']:
        path = (args.build / f'libmaya_expert_{backend}.so').resolve()
        manifest['libraries'][backend] = {'path': str(path), 'sha256': sha(path)}
        lib = c.CDLL(str(args.adapter.resolve() if args.adapter else path), mode=c.RTLD_LOCAL)
        fn = lib.maya_adapter_probe if args.adapter else lib.maya_expert_probe
        fn.argtypes = [c.c_uint32, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int,
            c.c_void_p, c.c_size_t, c.c_size_t, c.c_size_t, c.c_void_p, c.c_void_p,
            c.c_float, c.c_void_p, c.c_void_p, c.c_size_t]
        fn.restype = c.c_int
        if args.adapter:
            # Separate function objects prevent argument-type changes on a cached ctypes symbol.
            signature = c.CFUNCTYPE(c.c_int, c.c_char_p, *fn.argtypes)
            raw_function = signature(c.cast(fn, c.c_void_p).value)
            fn = lambda *a, _f=raw_function, _path=str(path).encode(): _f(_path, *a)
        functions[backend] = (lib, fn)  # keep the DSO loaded for the whole run

    contexts = {}
    if args.persistent:
        from persistent import Persistent
        for backend in functions:
            contexts[backend] = Persistent(functions[backend][0],
                args.cuda_device if backend == 'cuda' else args.hip_device,
                int(gate.tensor_type), int(down.tensor_type), stride)

    def call(backend, blob, routing, activation, **overrides):
        blob = np.ascontiguousarray(blob, dtype=np.uint8)
        routing = np.ascontiguousarray(routing, dtype=np.float32)
        activation = np.ascontiguousarray(activation, dtype=np.float32)
        buf = np.full(width + 2, -991, dtype=np.float32)
        err = c.create_string_buffer(512)
        rc = functions[backend][1](overrides.get('abi', 1),
            args.cuda_device if backend == 'cuda' else args.hip_device,
            int(gate.tensor_type), int(down.tensor_type), width, 2048, len(routing),
            blob.ctypes.data, overrides.get('bytes', blob.nbytes), stride, offset,
            routing.ctypes.data, activation.ctypes.data, 8.0, buf.ctypes.data + 4, err, len(err))
        if overrides:
            assert rc != 0, 'invalid descriptor was accepted'
            assert np.all(buf == -991), 'rejected call modified output'
            return
        if rc:
            raise RuntimeError(backend + ': ' + err.value.decode())
        assert buf[0] == -991 and buf[-1] == -991, 'output canary damaged'
        assert np.all(np.isfinite(buf[1:-1])), 'nonfinite result'
        reference = buf[1:-1].copy()
        if args.persistent:
            actual = contexts[backend](blob, routing, activation)
            assert np.array_equal(actual, reference), 'persistent/stateless mismatch'
            return actual
        return reference

    def metrics(a, b):
        delta = a.astype(np.float64) - b.astype(np.float64)
        return {'max_abs': float(np.max(np.abs(delta))),
            'relative_l2': float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30))}

    results, outputs = [], {}
    try:
        for token in [0, 1, 17, 63]:
            blob = np.concatenate([np.concatenate([a[e] for a in raw]) for e in ids[token]])
            manifest['fixture_blobs'][str(token)] = {'experts': ids[token].tolist(),
                'routing': routes[token].tolist(), 'sha256': hashlib.sha256(blob).hexdigest()}
            result = {'token': token}
            all_outputs = {}
            for backend in ['cuda', 'hip']:
                y = call(backend, blob, routes[token], x[token])
                repeat = call(backend, blob, routes[token], x[token])
                assert np.array_equal(y, repeat), 'nondeterministic repeated output'
                assert np.all(call(backend, blob, np.zeros(k, np.float32), x[token]) == 0), 'zero routes'
                assert np.all(call(backend, blob, routes[token], np.zeros(width, np.float32)) == 0), 'zero input'
                call(backend, blob, routes[token], x[token], abi=999)
                call(backend, blob, routes[token], x[token], bytes=blob.nbytes - 1)
                individual = [call(backend, blob[i*stride:(i+1)*stride], routes[token, i:i+1], x[token]) for i in range(k)]
                summed = np.sum(np.stack(individual), axis=0, dtype=np.float32)
                result[backend + '_single_vs_group'] = metrics(summed, y)
                # Same backend / same per-expert arithmetic, only sum grouping changes.
                assert np.allclose(summed, y, rtol=1e-4, atol=1e-5), 'weighted reduction inconsistency'
                all_outputs[backend] = y
                outputs[f'{token}_{backend}'] = y
            mixed = call('cuda', blob[:4*stride], routes[token, :4], x[token]) + call('hip', blob[4*stride:], routes[token, 4:], x[token])
            outputs[f'{token}_mixed'] = mixed
            result['hip_vs_cuda'] = metrics(all_outputs['hip'], all_outputs['cuda'])
            result['mixed_vs_cuda'] = metrics(mixed, all_outputs['cuda'])
            results.append(result)
            print(json.dumps(result), flush=True)
        if args.persistent:
            manifest['persistent_checks'] = {b:ctx.lifecycle_checks() for b,ctx in contexts.items()}
        manifest['structural_checks'] = 'PASS'
        manifest['quality_acceptance'] = 'NOT DECIDED; cross-backend errors reported, no full-model gate'
    finally:
        for ctx in contexts.values():
            ctx.close()
        manifest['results'] = results
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        np.savez(args.output / 'outputs.npz', **outputs)


if __name__ == '__main__':
    main()
