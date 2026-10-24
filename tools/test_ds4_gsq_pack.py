"""pack_gguf.py's pack of the ISTA-DASLab GSQ-RCO quants (DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS and the like):
the Engram file beside the shards is never a model shard, the metadata's alias spellings resolve, the Engram's hash
parameters arrive as integer tensors (engram_token_map I32, engram_primes / engram_offsets / engram_multipliers I64),
the Engram's q / k stay apart (engram.q_weight, engram.k_weight) with its kv as engram_wkv, a 3-D attn_output_a
splits in the output groups, and every failure is named (no silent partial packs).  Over minimal GGUFs written here -
no download, no model, no GPU.  The Engram q/k/wkv and hash-tensor handling is adapted from Strata 96ca4383
(MIT; see LICENSE.strata-gsq).

    python -m unittest tools.test_ds4_gsq_pack
"""

from __future__ import annotations

import json
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ds4"))
import pack_gguf as PG

GgufTensor = tuple[str, list[int], int, bytes]

PACK = ROOT / "tools" / "ds4" / "pack_gguf.py"

# the tiny model's geometry (one routed-expert layer, one Engram layer, two output groups)
VOCAB, DIM, FF, N_EXP, O_LORA, GROUPS = 9, 8, 8, 2, 4, 2
ENGRAM_LAYER, NGRAM, HEADS, HEAD_DIM = 1, 2, 2, 256
COLS = (NGRAM - 1) * HEADS
TOKEN_MAP = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # 3 compressed ids over the 9 tokens
PRIMES, OFFSETS, MULTS = [7, 11], [0, 7], [3, 5]  # [layer x cols] / [layer x ngram]


def gguf_string(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(name: str, value: str) -> bytes:
    return gguf_string(name) + struct.pack("<I", 8) + gguf_string(value)


def kv_u32(name: str, value: int) -> bytes:
    return gguf_string(name) + struct.pack("<I", 4) + struct.pack("<I", value)


def kv_f32(name: str, value: float) -> bytes:
    return gguf_string(name) + struct.pack("<I", 6) + struct.pack("<f", value)


def kv_i32_list(name: str, values: list[int]) -> bytes:
    return (
        gguf_string(name)
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 4, len(values))
        + struct.pack(f"<{len(values):d}i", *values)
    )


def kv_str_list(name: str, values: list[str]) -> bytes:
    return (
        gguf_string(name)
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 8, len(values))
        + b"".join(gguf_string(v) for v in values)
    )


def kv_f32_list(name: str, values: list[float]) -> bytes:
    return (
        gguf_string(name)
        + struct.pack("<I", 9)
        + struct.pack("<IQ", 6, len(values))
        + struct.pack(f"<{len(values):d}f", *values)
    )


def bf16_bytes(n: int) -> bytes:
    return b"".join(
        struct.pack("<H", 0x3F80 if i % 2 == 0 else 0x0000) for i in range(n)
    )


def write_gguf(
    path: pathlib.Path,
    kvs: list[bytes],
    tensors: list[tuple[str, list[int], int, bytes]],
) -> int:
    """A GGUF v3 of `tensors` (name, dims, ggml type, data), their data laid out 32-byte aligned in order.
    Returns the file's whole size."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    infos, off = b"", 0
    for name, dims, typ, data in tensors:
        infos += (
            gguf_string(name)
            + struct.pack("<I", len(dims))
            + struct.pack(f"<{len(dims):d}Q", *dims)
            + struct.pack("<IQ", typ, off)
        )
        off += (len(data) + 31) // 32 * 32
    head += infos
    head += b"\0" * ((-len(head)) % 32)
    body = b"".join(data + b"\0" * ((-len(data)) % 32) for _, _, _, data in tensors)
    path.write_bytes(head + body)
    return len(head + body)


def model_kv_bytes() -> list[bytes]:
    """The candidate's spellings: every Engram / candidate / source-layer key under its ISTA-DASLab name, prefixed
    with the architecture as a GGUF writes them."""
    a = "deepseek41."
    return [
        kv_str("general.architecture", "deepseek41"),
        kv_u32("split.count", 2),
        kv_str("general.name", "tiny-gsq"),
        kv_u32(a + "block_count", 2),
        kv_u32(a + "embedding_length", DIM),
        kv_u32(a + "attention.head_count", 2),
        kv_u32(a + "attention.key_length", 4),
        kv_u32(a + "rope.dimension_count", 4),
        kv_u32(a + "attention.q_lora_rank", 8),
        kv_u32(a + "attention.output_lora_rank", O_LORA),
        kv_u32(a + "attention.output_group_count", GROUPS),
        kv_u32(a + "expert_count", N_EXP),
        kv_u32(a + "expert_used_count", 1),
        kv_u32(a + "expert_feed_forward_length", FF),
        kv_u32(a + "attention.indexer.head_count", 2),
        kv_u32(a + "attention.indexer.key_length", 4),
        kv_u32(a + "attention.indexer.top_k", 2),
        kv_i32_list(a + "kv_source_layer_ids", [0]),
        kv_i32_list(a + "index_source_layer_ids", [1]),
        kv_u32(a + "attention.candidate_source_layer_id", 1),
        kv_u32(a + "attention.candidate_topk_blocks", 4),
        kv_u32(a + "attention.candidate_block_size", 2),
        kv_i32_list(a + "engram.layer_ids", [ENGRAM_LAYER]),
        kv_u32(a + "engram_max_ngram_size", NGRAM),
        kv_u32(a + "engram_n_heads", HEADS),
        kv_u32(a + "engram_head_dim", HEAD_DIM),
        kv_u32(a + "engram_pad_token_id", 2),
        kv_i32_list(a + "dspark_target_layer_ids", [1]),
        kv_f32_list(a + "swiglu_clamp_exp", [10.0, 10.0]),
        kv_f32(a + "rope.freq_base", 10000.0),
        kv_f32(a + "attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32(a + "hyper_connection.count", 1),
        kv_str_list("tokenizer.ggml.tokens", [chr(ord("a") + i) for i in range(VOCAB)]),
        kv_str_list("tokenizer.ggml.merges", []),
        kv_i32_list("tokenizer.ggml.token_type", [1] * VOCAB),
        kv_u32("tokenizer.ggml.bos_token_id", 0),
        kv_u32("tokenizer.ggml.eos_token_id", 1),
        kv_u32("tokenizer.ggml.padding_token_id", 2),
        kv_str(
            "tokenizer.chat_template",
            "{% for m in messages %}{{ m.content }}{% endfor %}",
        ),
    ]


def write_hash_shard(
    path: pathlib.Path, drop: str | None = None, extra: list[GgufTensor] | None = None
) -> int:
    """Shard 1: the Engram hash parameters as integer tensors (one dropped by `drop`)."""
    int_tensors = [
        (
            "engram_token_map.weight",
            [VOCAB],
            26,
            struct.pack(f"<{VOCAB:d}i", *TOKEN_MAP),
        ),
        ("engram_primes.weight", [COLS, 1], 27, struct.pack(f"<{COLS:d}q", *PRIMES)),
        ("engram_offsets.weight", [COLS, 1], 27, struct.pack(f"<{COLS:d}q", *OFFSETS)),
        (
            "engram_multipliers.weight",
            [NGRAM, 1],
            27,
            struct.pack(f"<{NGRAM:d}q", *MULTS),
        ),
    ]
    kept = [t for t in int_tensors if t[0] != drop]
    return write_gguf(path, model_kv_bytes(), kept + (extra or []))


def write_weight_shard(path: pathlib.Path) -> int:
    """Shard 2: the weights - global, a 3-D attn_output_a, an experts stack, and the Engram's q / k / wkv."""
    tensors = [
        ("token_embd.weight", [DIM, VOCAB], 30, bf16_bytes(DIM * VOCAB)),
        ("output_norm.weight", [DIM], 0, struct.pack(f"<{DIM:d}f", *([1.0] * DIM))),
        (
            "blk.0.attn_output_a.weight",
            [DIM // 2, O_LORA, GROUPS],
            30,
            bf16_bytes((DIM // 2) * O_LORA * GROUPS),
        ),
        ("blk.0.attn_output_b.weight", [DIM, O_LORA], 30, bf16_bytes(DIM * O_LORA)),
        (
            "blk.0.ffn_gate_exps.weight",
            [FF, DIM, N_EXP],
            30,
            bf16_bytes(FF * DIM * N_EXP),
        ),
        (f"blk.{ENGRAM_LAYER:d}.engram_q.weight", [DIM, 1], 30, bf16_bytes(DIM)),
        (f"blk.{ENGRAM_LAYER:d}.engram_k.weight", [DIM, 1], 30, bf16_bytes(DIM)),
        (
            f"blk.{ENGRAM_LAYER:d}.engram_wkv.weight",
            [DIM, DIM],
            30,
            bf16_bytes(DIM * DIM),
        ),
        ("output.weight", [DIM, VOCAB], 30, bf16_bytes(DIM * VOCAB)),
    ]
    return write_gguf(
        path,
        [kv_str("general.architecture", "deepseek41"), kv_u32("split.count", 2)],
        tensors,
    )


def engram_file(path: pathlib.Path, rows: int = 18, cut: int = 0) -> int:
    """The Engram tables' own file (architecture deepseek41-engram): one blk.N.engram_embd of IQ4_NL, its last
    `cut` bytes missing."""
    kvs = [
        kv_str("general.architecture", "deepseek41-engram"),
        kv_str("general.description", "Shared IQ4_NL Engram lookup tables"),
    ]
    data = bytes(rows * HEAD_DIM // 32 * 18)  # IQ4_NL: 18 bytes a block of 32
    tensors = [(f"blk.{ENGRAM_LAYER:d}.engram_embd.weight", [HEAD_DIM, rows], 20, data)]
    size = write_gguf(path, kvs, tensors)
    if cut:
        path.write_bytes(path.read_bytes()[: size - cut])
    return size - cut


def build_model(
    base: pathlib.Path,
    drop: str | None = None,
    engram: bool = True,
    engram_cut: int = 0,
    extra_a: list[GgufTensor] | None = None,
) -> pathlib.Path:
    m = base / "tiny-gsq"
    m.mkdir()
    write_hash_shard(m / "tiny-gsq-00001-of-00002.gguf", drop=drop, extra=extra_a)
    write_weight_shard(m / "tiny-gsq-00002-of-00002.gguf")
    if engram:
        engram_file(m / "tiny-engram-IQ4_NL.gguf", cut=engram_cut)
    return m


def run_pack(
    model: pathlib.Path, out: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PACK), "--model", str(model), "--out", str(out)],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )


def read_pack(
    out: pathlib.Path,
) -> tuple[dict[str, str], dict[str, tuple[str, int, int, int, list[int]]], list[str]]:
    geo, tensors, files = {}, {}, []
    for line in (out / "ds4.txt").read_text().splitlines():
        k, _, v = line.partition(" ")
        geo[k] = v
    for line in (out / "tensors.txt").read_text().splitlines():
        p = line.split(" ")
        if p[0] == "file":
            files.append(p[2])
        else:
            tensors[p[1]] = (
                p[2],
                int(p[3]),
                int(p[4]),
                int(p[5]),
                [int(x) for x in p[6:]],
            )
    return geo, tensors, files


class GsqPack(unittest.TestCase):
    """The whole pack of the synthetic GSQ-shaped model: artifacts, bytes, accounting, bounds."""

    def setUp(self) -> None:
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.base = pathlib.Path(self.tmp)
        self.model = build_model(self.base)
        self.out = self.base / "pack"
        r = run_pack(self.model, self.out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.geo, self.tensors, self.files = read_pack(self.out)

    def test_alias_metadata_reaches_ds4_txt(self) -> None:
        for k, want in (
            ("candidate_source_layer_id", "1"),
            ("candidate_topk_blocks", "4"),
            ("candidate_block_size", "2"),
            ("kv_source_layer_ids", "0"),
            ("index_source_layer_ids", "1"),
            ("engram_layer_ids", str(ENGRAM_LAYER)),
            ("dspark_target_layer_ids", str(ENGRAM_LAYER)),
            ("vocab_size", str(VOCAB)),
            ("num_hidden_layers", "2"),
            ("o_groups", str(GROUPS)),
            ("swiglu_limit", "10.0"),
        ):
            self.assertEqual(self.geo.get(k), want, k)
        self.assertEqual(self.geo.get("format"), "gguf")

    def test_engram_q_k_wkv_names_and_slices(self) -> None:
        t = self.tensors
        eg = f"layers.{ENGRAM_LAYER:d}."
        self.assertIn(eg + "engram.q_weight", t)
        self.assertIn(eg + "engram.k_weight", t)
        self.assertIn(eg + "engram.wkv.weight", t)
        self.assertNotIn(
            eg + "engram.gate.weight", t
        )  # (never folded here: the runtime multiplies q * k)
        for name in (eg + "engram.q_weight", eg + "engram.k_weight"):
            self.assertEqual(
                t[name][4], [1, DIM]
            )  # torch order: one hyper-connection row of the hidden size
        for gi in range(GROUPS):
            sl = t[f"layers.0.attn.wo_a.slice.{gi:d}.weight"]
            self.assertEqual(sl[4], [O_LORA, DIM // 2])
            # absolute in shard B: the data start, the two 32-aligned tensors before wo_a, then group g's rows
            aligned = lambda n: (n + 31) // 32 * 32
            before = aligned(DIM * VOCAB * 2) + aligned(DIM * 4)
            self.assertEqual(
                sl[2],
                PG.Gguf(self.model / "tiny-gsq-00002-of-00002.gguf").data_start
                + before
                + (DIM // 2) * O_LORA * 2 * gi,
            )
        for ei in range(N_EXP):
            self.assertEqual(t[f"layers.0.ffn.experts.{ei:d}.w1.weight"][4], [FF, DIM])

    def test_hash_tensors_are_metadata_not_weights(self) -> None:
        for name in (
            "engram_token_map.weight",
            "engram_primes.weight",
            "engram_offsets.weight",
            "engram_multipliers.weight",
        ):
            self.assertNotIn(name, self.tensors)
            self.assertNotIn("layers.0." + name, self.tensors)

    def test_every_consumed_tensor_is_accounted(self) -> None:
        """Every weight tensor of every source file appears under its checkpoint name exactly once, and their bytes
        sum to the sources' - only the hash tensors stay out."""
        want, src_bytes = set(), 0
        for n in (
            "tiny-gsq-00001-of-00002.gguf",
            "tiny-gsq-00002-of-00002.gguf",
            "tiny-engram-IQ4_NL.gguf",
        ):
            for name, typ, dims, off in PG.Gguf(self.model / n).tensors:
                if name in PG.ENGRAM_HASH_TENSORS:
                    continue
                _, per, bpb = PG.TYPES[typ]
                nel = 1
                for d in dims:
                    nel *= d
                src_bytes += nel // per * bpb
                m = re.match(r"blk\.(\d+)\.(.+)$", name)
                if m and m.group(2) in PG.LAYER:
                    base = f"layers.{m.group(1)!s}.{PG.LAYER[m.group(2)]!s}"
                    if base.endswith(".attn.wo_a.slice"):
                        want |= {base + f".{i:d}.weight" for i in range(GROUPS)}
                    elif ".ffn.experts." in base:
                        want |= {
                            f"{base.rsplit('.', 1)[0]!s}.{i:d}.w1.weight"
                            for i in range(N_EXP)
                        }
                    else:
                        want.add(base)
                elif name in PG.GLOBAL:
                    want.add(PG.GLOBAL[name])
        self.assertEqual(set(self.tensors), want)
        self.assertEqual(sum(t[3] for t in self.tensors.values()), src_bytes)

    def test_offsets_stay_within_their_files(self) -> None:
        for name, (dt, fi, off, nb, shape) in self.tensors.items():
            size = (self.model / self.files[fi]).stat().st_size
            self.assertLessEqual(off + nb, size, name)

    def test_engram_bin_bytes(self) -> None:
        b = (self.out / "engram.bin").read_bytes()
        want = struct.pack(
            "<4s6i", b"DSE1", VOCAB, 1, NGRAM, HEADS, HEAD_DIM, TOKEN_MAP[2]
        )
        want += struct.pack(f"<{VOCAB:d}i", *TOKEN_MAP)
        want += struct.pack(f"<{len(MULTS):d}q", *MULTS)
        want += struct.pack(f"<{len(PRIMES):d}q", *PRIMES)
        want += struct.pack(f"<{len(OFFSETS):d}q", *OFFSETS)
        self.assertEqual(b, want)

    def test_engram_file_referenced_in_place(self) -> None:
        self.assertIn("tiny-engram-IQ4_NL.gguf", self.files)
        self.assertIn("tiny-gsq-00001-of-00002.gguf", self.files)
        self.assertIn("tiny-gsq-00002-of-00002.gguf", self.files)

    def test_tokenizer_written(self) -> None:
        vocab = json.loads((self.out / "tokenizer" / "vocab.json").read_text())
        self.assertEqual(len(vocab), VOCAB)
        self.assertEqual(
            json.loads((self.out / "tokenizer" / "tokenizer.json").read_text())["pre"],
            "deepseek3",
        )


class EngramFileSharding(unittest.TestCase):
    """An Engram file (architecture deepseek41-engram or engram) beside the shards: never a model shard, never the
    metadata's source, yet its tables fill in the ones the shards leave out."""

    def setUp(self) -> None:
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.base = pathlib.Path(self.tmp)

    def test_is_engram_file_and_model_shard(self) -> None:
        m = self.base / "m.gguf"
        write_gguf(m, [kv_str("general.architecture", "deepseek41")], [])
        e1 = self.base / "e1.gguf"
        write_gguf(e1, [kv_str("general.architecture", "deepseek41-engram")], [])
        e2 = self.base / "e2.gguf"
        write_gguf(e2, [kv_str("general.architecture", "engram")], [])
        gm, ge1, ge2 = PG.Gguf(m), PG.Gguf(e1), PG.Gguf(e2)
        self.assertFalse(PG.is_engram_file(gm))
        self.assertTrue(PG.is_engram_file(ge1))
        self.assertTrue(PG.is_engram_file(ge2))
        for order in ([ge1, gm], [gm, ge1], [ge2, ge1, gm]):
            source = PG.model_shard(order)
            self.assertIs(source, gm)
        self.assertIsNone(PG.model_shard([ge1]))

    def test_explicit_model_shards_still_find_adjacent_engram(self) -> None:
        model = build_model(self.base)
        result = subprocess.run(
            [
                sys.executable,
                str(PACK),
                "--model",
                str(model),
                "--out",
                str(self.base / "pack"),
                "--shards",
                "tiny-gsq-00001-of-00002.gguf",
                "tiny-gsq-00002-of-00002.gguf",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        _, _, files = read_pack(self.base / "pack")
        self.assertEqual(len(files), 3)
        self.assertIn("tiny-engram-IQ4_NL.gguf", files)

    def test_a_split_counted_over_model_shards_only(self) -> None:
        """split.count 2 with a third (Engram) file in the folder still packs: the Engram file is no shard."""
        r = run_pack(build_model(self.base), self.base / "pack")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class FailClosed(unittest.TestCase):
    """A pack that cannot be built right fails with the field named - never a silent partial one."""

    def setUp(self) -> None:
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.base = pathlib.Path(self.tmp)
        self.n = 0

    def pack_fails(self, model: pathlib.Path, fragment: str) -> None:
        self.n += 1
        r = run_pack(model, self.base / (f"pack-{self.n:d}"))
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(fragment, r.stderr)

    def test_no_token_map(self) -> None:
        self.pack_fails(
            build_model(self.base, drop="engram_token_map.weight"), "engram.token_map"
        )

    def test_hash_parameter_counts(self) -> None:
        m = self.base / "m"
        m.mkdir()
        write_gguf(
            m / "a.gguf",
            model_kv_bytes(),
            [
                (
                    "engram_token_map.weight",
                    [VOCAB],
                    26,
                    struct.pack(f"<{VOCAB:d}i", *TOKEN_MAP),
                ),
                ("engram_primes.weight", [1, 1], 27, struct.pack("<q", 7)),
                (
                    "engram_offsets.weight",
                    [COLS, 1],
                    27,
                    struct.pack(f"<{COLS:d}q", *OFFSETS),
                ),
                (
                    "engram_multipliers.weight",
                    [NGRAM, 1],
                    27,
                    struct.pack(f"<{NGRAM:d}q", *MULTS),
                ),
            ],
        )
        write_weight_shard(m / "b.gguf")
        engram_file(m / "e.gguf")
        self.pack_fails(
            m, "engram: 2 multipliers, 1 primes, 2 offsets for 1 layers x 2 columns"
        )

    def test_missing_engram_tables(self) -> None:
        self.pack_fails(
            build_model(self.base, engram=False),
            f"blk.{ENGRAM_LAYER:d}.engram_embd.weight: missing",
        )

    def test_short_engram_file_in_the_model_folder(self) -> None:
        self.pack_fails(build_model(self.base, engram_cut=10), "not whole")

    def test_unmapped_tensor(self) -> None:
        self.pack_fails(
            build_model(
                self.base, extra_a=[("blk.0.mystery.weight", [4], 30, bf16_bytes(4))]
            ),
            "tensors the pack does not map",
        )

    def test_integer_tensor_with_an_unknown_name_is_no_metadata(self) -> None:
        self.pack_fails(
            build_model(
                self.base,
                extra_a=[
                    ("blk.0.mystery.weight", [4], 26, struct.pack("<4i", 1, 2, 3, 4))
                ],
            ),
            "which the engine does not read",
        )

    def test_wrong_hash_tensor_type(self) -> None:
        model = build_model(
            self.base,
            drop="engram_token_map.weight",
            extra_a=[("engram_token_map.weight", [VOCAB], 0, bytes(VOCAB * 4))],
        )
        self.pack_fails(
            model, "engram hash tensor engram_token_map.weight: expected ggml type 26"
        )

    def test_conflicting_hash_metadata(self) -> None:
        model = build_model(self.base)
        shard = PG.Gguf(model / "tiny-gsq-00001-of-00002.gguf")
        metadata = dict(shard.kv)
        metadata["deepseek41.engram.token_map"] = [1] * VOCAB
        with self.assertRaisesRegex(SystemExit, "conflicts with existing"):
            PG.int_tensors_to_engram([shard], metadata)

    def test_zero_prime_rejected_before_pack_files_exist(self) -> None:
        model = build_model(
            self.base,
            drop="engram_primes.weight",
            extra_a=[
                ("engram_primes.weight", [COLS, 1], 27, struct.pack("<2q", 0, 11))
            ],
        )
        self.pack_fails(model, "Engram primes must be positive")
        self.assertFalse((self.base / "pack-1" / "ds4.txt").exists())

    def test_hash_range_outside_engram_table(self) -> None:
        model = build_model(self.base)
        engram_file(model / "tiny-engram-IQ4_NL.gguf", rows=2)
        self.pack_fails(model, "Engram hash range exceeds table rows")

    def test_truncated_hash_payload(self) -> None:
        model = build_model(self.base)
        shard_path = model / "tiny-gsq-00001-of-00002.gguf"
        shard_path.write_bytes(shard_path.read_bytes()[:-32])
        self.pack_fails(
            model, "engram hash tensor engram_multipliers.weight: payload lies outside"
        )


class AliasLaw(unittest.TestCase):
    """Every alias spelling fills its geometry key in, and the GGUF's own key wins when it is there too -
    exhaustively over the alias table."""

    def test_original_wins_and_alias_fills_in(self) -> None:
        for key, alts in PG.ALIASES.items():
            v = [7] if key.endswith(("layers", "source_layers", "target_layers")) else 7
            for alt in alts:
                kv = {"general.architecture": "deepseek41", f"deepseek41.{alt!s}": v}
                filled = PG.geometry(dict(kv), 9)
                bare = PG.geometry({"general.architecture": "deepseek41"}, 9)
                changed = [
                    k for k in set(filled) | set(bare) if filled.get(k) != bare.get(k)
                ]
                self.assertEqual(len(changed), 1, key)
                self.assertEqual(filled[changed[0]], v, key)
                canonical = [3] if isinstance(v, list) else 3
                both = PG.geometry({**kv, f"deepseek41.{key!s}": canonical}, 9)
                self.assertEqual(both[changed[0]], canonical, key)


class EngramBinLaw(unittest.TestCase):
    """engram.bin round-trips its inputs, over every small (vocab, layers, ngram, heads) - the layout the engine's
    DSE1 reader walks (magic, 6 int32, the map, multipliers, primes, offsets)."""

    def round_trip(
        self, vocab: int, nl: int, ngram: int, heads: int, tm_len: int
    ) -> tuple[bytes, tuple[int, int, int, int]]:
        cols = (ngram - 1) * heads
        kv = {
            "general.architecture": "deepseek41",
            "deepseek41.engram.layers": [10 * i for i in range(nl)],
            "deepseek41.engram.max_ngram_size": ngram,
            "deepseek41.engram.head_count": heads,
            "deepseek41.engram.head_dim": 256,
            "deepseek41.engram.pad_token_id": 1,
            "deepseek41.engram.token_map": list(range(tm_len)),
            "deepseek41.engram.multipliers": list(range(1000, 1000 + nl * ngram)),
            "deepseek41.engram.primes": list(range(2000, 2000 + nl * cols)),
            "deepseek41.engram.offsets": list(range(3000, 3000 + nl * cols)),
        }
        out = pathlib.Path(self.tmp)
        nl_got, cols_got = PG.write_engram(out / "e.bin", kv, vocab)
        self.assertEqual((nl_got, cols_got), (nl, cols))
        return (out / "e.bin").read_bytes(), (nl, ngram, heads, cols)

    def setUp(self) -> None:
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())

    def test_round_trip_over_the_whole_grid(self) -> None:
        for vocab in (5, 9):
            for nl in (1, 2, 3):
                for ngram in (2, 3, 4):
                    for heads in (1, 2):
                        b, (_nl2, ng2, _nh2, cols) = self.round_trip(
                            vocab, nl, ngram, heads, vocab - 1
                        )
                        magic, v, l2, g2, h2, hd2, pad = struct.unpack("<4s6i", b[:28])
                        self.assertEqual(magic, b"DSE1")
                        self.assertEqual(
                            (v, l2, g2, h2, hd2, pad), (vocab, nl, ngram, heads, 256, 1)
                        )
                        pos = 28
                        self.assertEqual(
                            list(
                                struct.unpack(f"<{vocab:d}i", b[pos : pos + 4 * vocab])
                            ),
                            list(range(vocab - 1)) + [0],
                        )  # (the id past the map's pads to 0)
                        pos += 4 * vocab
                        self.assertEqual(
                            list(
                                struct.unpack(
                                    f"<{nl * ng2:d}q", b[pos : pos + 8 * nl * ng2]
                                )
                            ),
                            list(range(1000, 1000 + nl * ngram)),
                        )
                        pos += 8 * nl * ng2
                        self.assertEqual(
                            list(
                                struct.unpack(
                                    f"<{nl * cols:d}q", b[pos : pos + 8 * nl * cols]
                                )
                            ),
                            list(range(2000, 2000 + nl * cols)),
                        )
                        pos += 8 * nl * cols
                        self.assertEqual(
                            list(
                                struct.unpack(
                                    f"<{nl * cols:d}q", b[pos : pos + 8 * nl * cols]
                                )
                            ),
                            list(range(3000, 3000 + nl * cols)),
                        )
                        pos += 8 * nl * cols
                        self.assertEqual(pos, len(b))

    def test_fails_closed_without_a_token_map(self) -> None:
        kv = {
            "general.architecture": "deepseek41",
            "deepseek41.engram.layers": [1],
            "deepseek41.engram.max_ngram_size": 2,
            "deepseek41.engram.head_count": 2,
            "deepseek41.engram.head_dim": 256,
            "deepseek41.engram.pad_token_id": 1,
            "deepseek41.engram.multipliers": [3, 5],
            "deepseek41.engram.primes": [7, 11],
            "deepseek41.engram.offsets": [0, 7],
        }
        with self.assertRaises(SystemExit):
            PG.write_engram(pathlib.Path(self.tmp) / "e.bin", kv, 9)

    def test_fails_closed_when_the_map_exceeds_the_vocab(self) -> None:
        kv = {
            "general.architecture": "deepseek41",
            "deepseek41.engram.layers": [1],
            "deepseek41.engram.max_ngram_size": 2,
            "deepseek41.engram.head_count": 2,
            "deepseek41.engram.head_dim": 256,
            "deepseek41.engram.pad_token_id": 1,
            "deepseek41.engram.token_map": [0, 1, 2, 3],
            "deepseek41.engram.multipliers": [3, 5],
            "deepseek41.engram.primes": [7, 11],
            "deepseek41.engram.offsets": [0, 7],
        }
        with self.assertRaises(SystemExit):
            PG.write_engram(pathlib.Path(self.tmp) / "e.bin", kv, 2)

    def test_fails_closed_when_the_pad_id_is_out_of_the_map(self) -> None:
        kv = {
            "general.architecture": "deepseek41",
            "deepseek41.engram.layers": [1],
            "deepseek41.engram.max_ngram_size": 2,
            "deepseek41.engram.head_count": 2,
            "deepseek41.engram.head_dim": 256,
            "deepseek41.engram.pad_token_id": 9,
            "deepseek41.engram.token_map": [0, 1, 2],
            "deepseek41.engram.multipliers": [3, 5],
            "deepseek41.engram.primes": [7, 11],
            "deepseek41.engram.offsets": [0, 7],
        }
        with self.assertRaises(SystemExit):
            PG.write_engram(pathlib.Path(self.tmp) / "e.bin", kv, 9)


class OldMayaPacks(unittest.TestCase):
    """The pre-GSQ path stands: dotted metadata wins over every alias, and a quant without the tables still takes
    them from an Engram/ folder beside it."""

    def test_dotted_metadata_stands(self) -> None:
        kv = {
            "general.architecture": "deepseek41",
            "deepseek41.attention.kv_source_layers": [3],
            "deepseek41.kv_source_layer_ids": [9],
            "deepseek41.engram.layers": [5],
            "deepseek41.engram.layer_ids": [6],
        }
        geo = PG.geometry(kv, 9)
        self.assertEqual(geo["kv_source_layer_ids"], [3])
        self.assertEqual(geo["engram_layer_ids"], [5])

    def test_engram_set_folder_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            model = base / "Maya-S"
            model.mkdir()
            self.assertEqual(PG.engram_set(model, None), (None, []))
            folder = base / "Engram"
            folder.mkdir()
            self.assertEqual(
                PG.engram_set(model, None), (None, [])
            )  # (an empty folder is passed over)
            engram_file(folder / "t.gguf")
            got_dir, got = PG.engram_set(model, None)
            self.assertEqual(got_dir, folder)
            self.assertEqual(len(got), 1)
            self.assertEqual(
                got[0].tensors[0][0], f"blk.{ENGRAM_LAYER:d}.engram_embd.weight"
            )

    def test_explicit_engram_names_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
            engram_file(f)
            got_dir, got = PG.engram_set(pathlib.Path(d) / "some-model", str(f))
            self.assertEqual(got_dir, f)
            self.assertEqual(len(got), 1)


if __name__ == "__main__":
    unittest.main()
