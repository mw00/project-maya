"""Offline HTTP vision-cache regression replay without a GPU or model weights.

Real Service and Vision code use a disposable protocol encoder. The mock model
checks every assembled record before answering. The output directory keeps a JSON
receipt, encoder trace and server log for review.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from unittest.mock import patch


ENCODER = r'''
import json, os, struct, sys, time
from pathlib import Path
seen = {}
print("READY", flush=True)
for line in sys.stdin:
    parts = line.strip().split()
    if parts == ["QUIT"]:
        break
    kind, source, output = parts
    number = struct.unpack("<I", Path(source).read_bytes()[-4:])[0]
    seen[number] = seen.get(number, 0) + 1
    with open(os.environ["MAYA_TEST_ENCODER_TRACE"], "a") as trace:
        trace.write(json.dumps({"kind": kind, "number": number}) + "\n")
    if 790000 <= number < 790100:
        time.sleep(0.05)
    if number == 800000:
        print("ERR owned test encoder rejection", flush=True)
        continue
    if not (900000 <= number < 900100 and seen[number] == 1):
        counts = [1, 2] if kind == "VID" else [1]
        with open(output, "wb") as target:
            for frame, n in enumerate(counts):
                target.write(struct.pack("<5i", 0x31455653, n, 1, 1, 4))
                target.write(struct.pack("<4f", float(number + frame), 0., 0., 1.) * n)
    print("OK 1", flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maya-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(args.maya_root), str(args.maya_root / 'tools')]
    from serve.frontend import ChatTemplate
    from serve.server import ByteTokenizer, MockEngine, Service, Vision, serve

    expected, records, outcomes = {}, {}, []
    gate_a, gate_b = threading.Event(), threading.Event()

    class Tokenizer(ByteTokenizer):
        SPECIALS = ByteTokenizer.SPECIALS + ['<|begin_of_image|>', '<|image|>', '<|end_of_image|>',
                                            '<|begin_of_video|>', '<|video|>', '<|end_of_video|>']

        def __init__(self):
            self.local = threading.local()

        def encode(self, text, parse_special=False, plain=()):
            if (text == '<|image|>' and not parse_special and
                    getattr(self.local, 'case', '') == 'concurrent-a' and not gate_a.is_set()):
                # Both versions reach this after releasing the encoding FIFO turn.
                # A different request then evicts A's cached source before A continues.
                gate_a.set()
                if not gate_b.wait(10):
                    raise RuntimeError('concurrent preparation did not finish')
            return super().encode(text, parse_special=parse_special, plain=plain)

    tok = Tokenizer()

    class CheckingEngine(MockEngine):
        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            case = re.search(r'CASE=([a-z0-9-]+)', tok.decode(ids)).group(1)
            found = []
            with open(embeddings, 'rb') as source:
                while header := source.read(20):
                    magic, n, _, _, width = struct.unpack('<5i', header)
                    assert magic == 0x31455653 and width == 4
                    payload = source.read(n * width * 4)
                    assert len(payload) == n * width * 4
                    found.append([int(struct.unpack('<f', payload[:4])[0]), n])
            assert found == expected[case], (case, found, expected[case])
            records[case] = found
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    class TaggedService(Service):
        def prepare(self, messages, *positional, **kwargs):
            tok.local.case = re.search(r'CASE=([a-z0-9-]+)', json.dumps(messages)).group(1)
            return super().prepare(messages, *positional, **kwargs)

    encoder = args.out / 'encoder.py'
    encoder.write_text(ENCODER)
    trace = args.out / 'encoder-calls.jsonl'
    vision = Vision({'exe': sys.executable, 'mmproj': 'unused', 'model': 'unused'}, start=False,
                    env={**os.environ, 'MAYA_TEST_ENCODER_TRACE': str(trace)})
    vision.args = [sys.executable, '-u', str(encoder)]
    vision._start()
    engine = CheckingEngine(tok, 'ok', max_context=262144)
    svc = TaggedService(engine, tok, ChatTemplate(args.maya_root / 'serve/fixtures/glm_tool_images.jinja'),
                        vision=vision)
    httpd = serve(svc, port=0)
    base = f'http://127.0.0.1:{httpd.server_address[1]}'

    def media(number, video=False, api='openai'):
        raw = (b'\0\0\0\x18ftypisom' if video else b'\x89PNG\r\n\x1a\n') + struct.pack('<I', number)
        mime = 'video/mp4' if video else 'image/png'
        data = base64.b64encode(raw).decode()
        if api == 'anthropic':
            return {'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': data}}
        if api == 'responses':
            return {'type': 'input_image', 'image_url': f'data:{mime};base64,{data}'}
        kind = 'video_url' if video else 'image_url'
        return {'type': kind, kind: {'url': f'data:{mime};base64,{data}'}}

    def request(case, numbers, *, api='openai', stream=False, video=False):
        expected[case] = [[number, 1] for number in numbers]
        if video:
            expected[case] = [[numbers[0], 1], [numbers[0] + 1, 2]]
        text = {'type': 'input_text' if api == 'responses' else 'text', 'text': 'CASE=' + case}
        parts = [text] + [media(n, video, api) for n in numbers]
        body = {'model': 'm', 'messages': [{'role': 'user', 'content': parts}], 'max_tokens': 32,
                'stream': stream, 'reasoning_effort': 'none'}
        route = '/v1/chat/completions'
        if api == 'anthropic':
            route = '/v1/messages'
        elif api == 'responses':
            route = '/v1/responses'
            body = {'model': 'm', 'input': [{'role': 'user', 'content': parts}], 'max_output_tokens': 32}
        req = urllib.request.Request(base + route, json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                raw = response.read().decode()
                return response.status, raw
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def successful(case, numbers, **kwargs):
        status, reply = request(case, numbers, **kwargs)
        assert status == 200, (status, reply)
        assert records[case] == expected[case]
        assert len(vision.cache) <= 64
        return {'status': status, 'records': len(records[case]), 'cache_entries': len(vision.cache)}

    def stale(case, number, *, video=False):
        successful(case + '-first', [number], video=video)
        number_path = next(path for path, _ in vision.cache.values()
                           if path.is_file() and int(struct.unpack('<f', path.read_bytes()[20:24])[0]) == number)
        number_path.unlink()
        return successful(case + '-retry', [number], video=video)

    def unavailable(api, stream=False):
        number = 900010 + {'openai': 0, 'anthropic': 2, 'responses': 3}[api] + int(stream)
        case = 'missing-' + api + ('-stream' if stream else '')
        before = {p.name for p in vision.dir.glob('req-*.sve')}
        status, raw = request(case, [number], api=api, stream=stream)
        body = json.loads(raw)
        assert status == 503 and body.get('error'), (status, body)
        assert {p.name for p in vision.dir.glob('req-*.sve')} == before
        result = successful(case + '-retry', [number], api=api, stream=stream)
        return {'failure_status': status, 'retry': result}

    def rejection():
        before = {p.name for p in vision.dir.glob('req-*.sve')}
        status, reply = request('encoder-rejected', [42, 800000])
        assert status == 400 and json.loads(reply).get('error'), (status, reply)
        assert {p.name for p in vision.dir.glob('req-*.sve')} == before
        return {'status': status, 'partial_request_removed': True}

    def concurrent():
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(successful, 'concurrent-a', [70000])
            assert gate_a.wait(10), 'A did not reach the post-encode boundary'
            try:
                b = pool.submit(successful, 'concurrent-b', list(range(71000, 71064)))
                result_b = b.result(timeout=15)
            finally:
                gate_b.set()
            result_a = a.result(timeout=15)
        return {'a': result_a, 'b': result_b, 'forced_interleaving': True}

    def cancellation():
        before = {p.name for p in vision.dir.glob('req-*.sve')}
        parts = [{'type': 'input_text', 'text': 'CASE=cancelled-preparation'}]
        parts += [media(n, api='responses') for n in range(790000, 790080)]
        body = {'model': 'm', 'input': [{'role': 'user', 'content': parts}],
                'max_output_tokens': 32, 'background': True, 'store': True}

        def api(route, payload=None):
            req = urllib.request.Request(base + route, None if payload is None else json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.load(response)

        job = api('/v1/responses', body)
        deadline = time.monotonic() + 5
        while '"number": 790000' not in trace.read_text() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert '"number": 790000' in trace.read_text(), 'job never started encoding'
        cancelled = api('/v1/responses/' + job['id'] + '/cancel', {})
        assert cancelled['status'] == 'cancelled', cancelled
        with vision.lock:  # The active preparation must release its partial file before this succeeds.
            assert {p.name for p in vision.dir.glob('req-*.sve')} == before
        assert 'cancelled-preparation' not in records
        assert sum(790000 <= json.loads(line)['number'] < 790100 for line in trace.read_text().splitlines()) < 80
        successful('after-cancellation', [790200])
        return {'status': cancelled['status'], 'partial_request_removed': True, 'next_request_passed': True}

    def run_case(name, callback):
        try:
            detail = callback()
            outcomes.append({'case': name, 'passed': True, 'detail': detail})
        except Exception as error:
            outcomes.append({'case': name, 'passed': False, 'error': type(error).__name__ + ': ' + str(error)})

    try:
        with (args.out / 'server-output.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run_case('64-image-control', lambda: successful('images-64', list(range(64))))
            run_case('full-cache-two-image-followup', lambda: successful('warm-two-images', [0, 64]))
            run_case('65-images', lambda: successful('images-65', list(range(65))))
            run_case('130-images', lambda: successful('images-130', list(range(130))))
            run_case('order-and-duplicates', lambda: successful('ordered', [200] + list(range(201, 266)) + [200, 203]))
            run_case('stale-image-cache', lambda: stale('stale-image', 50000))
            run_case('concurrent-preparation', concurrent)
            run_case('missing-output-openai', lambda: unavailable('openai'))
            run_case('missing-output-openai-stream', lambda: unavailable('openai', True))
            run_case('missing-output-anthropic', lambda: unavailable('anthropic'))
            run_case('missing-output-responses', lambda: unavailable('responses'))
            run_case('encoder-rejection-cleanup', rejection)
            real_which = shutil.which

            def fixture_which(command, *positional, **keywords):
                # The protocol fixture produces frames without FFmpeg. Duration may be unknown.
                if command in ('ffmpeg', 'ffprobe'):
                    return sys.executable
                return real_which(command, *positional, **keywords)

            with patch('shutil.which', side_effect=fixture_which):
                run_case('stale-video-cache', lambda: stale('stale-video', 60000, video=True))
            run_case('cancel-during-preparation', cancellation)
    finally:
        gate_b.set()
        httpd.shutdown()
        httpd.server_close()
        svc.response_jobs.close()
        vision.close()
        shutil.rmtree(vision.dir)
    result = {'passed': all(row['passed'] for row in outcomes), 'cases': outcomes,
              'provider_source': str(args.maya_root / 'serve/server.py'),
              'provider_sha256': hashlib.sha256((args.maya_root / 'serve/server.py').read_bytes()).hexdigest(),
              'real_http_service_and_vision': True, 'fake_protocol_encoder_and_model': True,
              'no_gpu_weights_or_live_requests': True, 'model_verified_record_order_and_counts': True,
              'fixture_encoder_directory_removed': True}
    (args.out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
