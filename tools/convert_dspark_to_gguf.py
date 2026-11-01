#!/usr/bin/env python3
"""A DSpark drafter (LithosAI's Qwen3.6-35B-A3B-DSpark-NVFP4: DeepSpec / lithos-metal naming, ModelOpt NVFP4 storage)
-> an F16 GGUF (arch `qwen35-dspark-draft`) in the tensor names strata-dense's drafter loader reads.

NVFP4: E2M1 codes two a byte (low nibble first), an F8_E4M3 scale per 16 inputs, an F32 tensor scale;
value = E2M1(code) * E4M3(block scale) * tensor scale.  BF16 tensors as they are.  Reads the safetensors directly.

    python convert_dspark_to_gguf.py <checkpoint dir> <out.gguf>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_writer import GGUFWriter, _Tensor  # noqa: E402

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3 = e4m3_table()


class Shards:
    def __init__(self, d: Path):
        self.t = {}
        for f in sorted(d.glob("*.safetensors")):
            raw = open(f, "rb").read()
            n = struct.unpack("<Q", raw[:8])[0]
            h = json.loads(raw[8:8 + n])
            base = 8 + n
            for k, v in h.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.t[k] = (v["dtype"], v["shape"], raw, base + a, base + b)

    def has(self, k):
        return k in self.t

    def get(self, k):
        dt, shape, raw, a, b = self.t[k]
        buf = memoryview(raw)[a:b]
        if dt == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if dt == "F32":
            return np.frombuffer(buf, dtype=np.float32).reshape(shape)
        if dt == "U8" or dt == "F8_E4M3":
            return np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        raise ValueError(f"{k}: dtype {dt}")

    def weight(self, k):
        """a matrix [out, in] as float32: NVFP4 dequantized, or BF16"""
        if self.t[k + ".weight"][0] != "U8":
            return self.get(k + ".weight")
        codes = self.get(k + ".weight")                               # [out, in / 2]
        scale = E4M3[self.get(k + ".weight_scale")]                   # [out, in / 16]
        ts = float(self.get(k + ".weight_scale_2").reshape(-1)[0])
        out = codes.shape[0]
        lo, hi = E2M1[codes & 0xF], E2M1[codes >> 4]
        v = np.empty((out, codes.shape[1] * 2), dtype=np.float32)
        v[:, 0::2], v[:, 1::2] = lo, hi
        v *= np.repeat(scale, 16, axis=1) * ts
        return v


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = json.load(open(src / "config.json"))
    st = Shards(src)
    w = GGUFWriter()
    A = "qwen35-dspark-draft"
    taps = cfg.get("target_layer_ids") or cfg["dflash_config"]["target_layer_ids"]
    w.add("general.architecture", A)
    w.add("general.name", "DSpark-Draft")
    for k, v in [("block_count", cfg["num_hidden_layers"]), ("embedding_length", cfg["hidden_size"]),
                 ("feed_forward_length", cfg["intermediate_size"]), ("attention.head_count", cfg["num_attention_heads"]),
                 ("attention.head_count_kv", cfg["num_key_value_heads"]), ("attention.key_length", cfg["head_dim"]),
                 ("attention.value_length", cfg["head_dim"]), ("vocab_size", cfg["vocab_size"]),
                 ("dflash.block_size", cfg.get("block_size", 8)), ("dflash.mask_token_id", cfg["mask_token_id"]),
                 ("dspark.markov_rank", cfg["markov_rank"])]:
        w.add(f"{A}.{k}", int(v), "u32")
    w.add(f"{A}.attention.layer_norm_rms_epsilon", float(cfg["rms_norm_eps"]), "f32")
    w.add(f"{A}.rope.freq_base", float(cfg["rope_parameters"]["rope_theta"]), "f32")
    w.add(f"{A}.dflash.target_layer_ids", [int(x) for x in taps], "array:i32")

    def put(name, arr, tname, dt):   # GGUF order: dim 0 the fastest (a row-major [out, in] -> [in, out])
        a = np.ascontiguousarray(arr.astype(dt))
        w.tensors.append(_Tensor(name, list(reversed(a.shape)), tname, a.tobytes()))

    def f16(name, arr):
        put(name, arr, "F16", "<f2")

    def f32(name, arr):
        put(name, arr, "F32", "<f4")

    f16("token_embd.weight", st.weight("embed_tokens"))
    f16("output.weight", st.weight("lm_head"))
    f16("dflash.fc.weight", st.weight("fc"))
    f32("dflash.hidden_norm.weight", st.get("hidden_norm.weight"))
    f32("output_norm.weight", st.get("norm.weight"))
    f16("dspark.markov_w1", st.get("markov_head.markov_w1.weight"))
    f16("dspark.markov_w2.weight", st.weight("markov_head.markov_w2"))
    for i in range(cfg["num_hidden_layers"]):
        p, q = f"layers.{i}.", f"blk.{i}."
        f32(q + "attn_norm.weight", st.get(p + "input_layernorm.weight"))
        f32(q + "ffn_norm.weight", st.get(p + "post_attention_layernorm.weight"))
        f32(q + "attn_q_norm.weight", st.get(p + "self_attn.q_norm.weight"))
        f32(q + "attn_k_norm.weight", st.get(p + "self_attn.k_norm.weight"))
        for a, b in [("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")]:
            f16(q + b + ".weight", st.weight(p + "self_attn." + a))
        for a, b in [("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")]:
            f16(q + b + ".weight", st.weight(p + "mlp." + a))
        print("layer", i, "done", flush=True)
    w.write(dst)
    print("wrote", dst)


if __name__ == "__main__":
    main()
