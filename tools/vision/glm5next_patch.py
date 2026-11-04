"""tools/vision/glm5next_patch.py - GLM-5.3-Flash's vision tower in llama.cpp's mtmd, at the commit setup pins.

The projector type "glm5next" (mmproj from glm5next_mmproj.py) next to GLM-4V's, whose shape it shares: the same
tensors, merge-window patch order, 2D positions, <|begin_of_image|> / <|end_of_image|> and dynamic-size
preprocessing; its own graph (glm5next/glm5next.cpp) and a SwiGLU clamp in build_ffn (hparams.ffn_clamp).
Every edit is anchored on the pinned source and idempotent: a second run changes nothing, and an anchor that is
missing (another llama.cpp) stops with the file it was looking in.

    python glm5next_patch.py <llama.cpp dir>
"""
from __future__ import annotations

import pathlib
import shutil
import sys

MARK = "glm5next"


def edit(path: pathlib.Path, pairs: list[tuple[str, str]], count: int | None = 1):
    s = path.read_text(encoding="utf-8")
    for old, new in pairs:
        if new in s:
            continue                                   # already applied
        n = s.count(old)
        if n == 0 or (count is not None and n != count):
            raise SystemExit(f"glm5next patch: {path}: anchor found {n} times:\n{old}")
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8", newline="\n")


def main():
    root = pathlib.Path(sys.argv[1])
    m = root / "tools" / "mtmd"
    here = pathlib.Path(__file__).resolve().parent

    edit(m / "clip-impl.h", [
        ("    PROJECTOR_TYPE_GLM4V,\n", "    PROJECTOR_TYPE_GLM4V,\n    PROJECTOR_TYPE_GLM5NEXT,\n"),
        ('    { PROJECTOR_TYPE_GLM4V,             "glm4v"},\n',
         '    { PROJECTOR_TYPE_GLM4V,             "glm4v"},\n    { PROJECTOR_TYPE_GLM5NEXT,          "glm5next"},\n'),
    ])
    edit(m / "clip-model.h", [
        ("    ffn_op_type ffn_op = FFN_GELU;\n",
         "    ffn_op_type ffn_op = FFN_GELU;\n"
         "    float ffn_clamp = 0.0f;   // glm5next: the SwiGLU gate clamped from above, up both ways (0 = off)\n"),
    ])
    edit(m / "models" / "models.h", [
        ("struct clip_graph_glm4v : clip_graph {\n"
         "    clip_graph_glm4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n",
         "struct clip_graph_glm4v : clip_graph {\n"
         "    clip_graph_glm4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n\n"
         "struct clip_graph_glm5next : clip_graph {\n"
         "    clip_graph_glm5next(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n"),
    ])
    edit(m / "CMakeLists.txt", [
        ("            models/glm4v.cpp\n", "            models/glm4v.cpp\n            models/glm5next.cpp\n"),
    ])
    shutil.copy2(here / "glm5next" / "glm5next.cpp", m / "models" / "glm5next.cpp")

    clip = m / "clip.cpp"
    edit(clip, [
        # the SwiGLU clamp (glm5next's blocks and merger: gate <= limit, -limit <= up <= limit)
        ("        case FFN_SILU:\n"
         "            if (gate) {\n"
         "                cur = ggml_swiglu_split(ctx0, cur, tmp);\n",
         "        case FFN_SILU:\n"
         "            if (gate) {\n"
         "                if (hparams.ffn_clamp > 0.0f) {   // glm5next\n"
         "                    cur = ggml_clamp(ctx0, cur, -1e30f, hparams.ffn_clamp);\n"
         "                    tmp = ggml_clamp(ctx0, tmp, -hparams.ffn_clamp, hparams.ffn_clamp);\n"
         "                }\n"
         "                cur = ggml_swiglu_split(ctx0, cur, tmp);\n"),
        # the graph
        ("        case PROJECTOR_TYPE_GLM4V:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_glm4v>(ctx, img);\n"
         "            } break;\n",
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_glm4v>(ctx, img);\n"
         "            } break;\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_glm5next>(ctx, img);\n"
         "            } break;\n"),
        # hyperparameters: GLM-4V's, the processor's token limits, the clamp
        ("                case PROJECTOR_TYPE_GLM4V:\n"
         "                    {\n"
         "                        hparams.rope_theta = 10000.0f;\n",
         "                case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                    {\n"
         "                        hparams.rope_theta = 10000.0f;\n"
         "                        hparams.n_merge = 2;\n"
         "                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;\n"
         "                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);\n"
         "                        int min_tok = 16, max_tok = 8000;\n"
         "                        get_u32(\"clip.vision.image_min_tokens\", min_tok, false);\n"
         "                        get_u32(\"clip.vision.image_max_tokens\", max_tok, false);\n"
         "                        hparams.set_limit_image_tokens(min_tok, max_tok);\n"
         "                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup\n"
         "                        get_f32(\"clip.vision.ffn_clamp\", hparams.ffn_clamp, false);\n"
         "                    } break;\n"
         "                case PROJECTOR_TYPE_GLM4V:\n"
         "                    {\n"
         "                        hparams.rope_theta = 10000.0f;\n"),
        # the projector tensors are GLM-4V's
        ("            case PROJECTOR_TYPE_GLM4V:\n"
         "                {\n"
         "                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, \"weight\"));\n",
         "            case PROJECTOR_TYPE_GLM4V:\n"
         "            case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                {\n"
         "                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, \"weight\"));\n"),
        # embedding size
        ("        case PROJECTOR_TYPE_GLM4V:\n"
         "            return ctx->model.mm_ffn_down_w->ne[1];\n",
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            return ctx->model.mm_ffn_down_w->ne[1];\n"),
        # 2D positions (merge-window order)
        ("        case PROJECTOR_TYPE_QWEN3VL:\n"
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "            {\n"
         "                const int merge_ratio = hparams.n_merge;\n",
         "        case PROJECTOR_TYPE_QWEN3VL:\n"
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "            {\n"
         "                const int merge_ratio = hparams.n_merge;\n"),
        # patch count (two conv frames)
        ("        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_YOUTUVL:\n"
         "        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "            {\n"
         "                // dynamic size (2 conv, so double patch size)\n",
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "        case PROJECTOR_TYPE_YOUTUVL:\n"
         "        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "            {\n"
         "                // dynamic size (2 conv, so double patch size)\n"),
    ])
    # output grid width and height (two switches with the same label list)
    edit(clip, [
        ("        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_PADDLEOCR:\n",
         "        case PROJECTOR_TYPE_GLM4V:\n"
         "        case PROJECTOR_TYPE_GLM5NEXT:\n"
         "        case PROJECTOR_TYPE_PADDLEOCR:\n"),
    ], count=None)
    edit(m / "mtmd.cpp", [
        ("            case PROJECTOR_TYPE_GLM4V:\n"
         "                {\n"
         "                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>\n",
         "            case PROJECTOR_TYPE_GLM4V:\n"
         "            case PROJECTOR_TYPE_GLM5NEXT:\n"
         "                {\n"
         "                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>\n"),
    ])
    print(f"llama.cpp at {root}: {MARK} vision added")
    # the encoder's float32 sums on NVIDIA and AMD GPUs (compiled only into maya-vision: the engines' ggml is the
    # same code as before)
    sys.path.insert(0, str(here))
    import ggml_vision_precision
    ggml_vision_precision.main(root)


if __name__ == "__main__":
    main()
