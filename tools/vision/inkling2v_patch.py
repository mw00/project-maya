"""tools/vision/inkling2v_patch.py - Inkling 2's vision tower in llama.cpp's mtmd, at the commit setup pins.

The projector type "inkling2v" (unsloth's mmproj-BF16.gguf of Inkling-2-Mini) from upstream llama.cpp commit
5c008e000 ("feat: add Inkling 2 support (text + vision)", #32217), only its vision encoder: the graph
(inkling2v/inkling2v.cpp), its hyperparameters, tensors, token counts, 2D positions (the CLS token at 0, the patches
from 1) and its preprocessor (padded bottom-right to a multiple of patch x merge, resized only above the token cap).
Upstream later moved these files (clip.cpp -> mtmd-model.cpp / mtmd-graph.cpp / mtmd-subcontext.cpp); the code is
the same, in the pinned commit's names.  Neither the text model nor ggml is touched.

maya-vision reads Inkling's pictures without a text model (the pinned libllama has no "inkling2" architecture to open
its GGUF with, and the encoder needs only the mmproj): the server writes the template's markers
(<|content_image|><|unused_200054|>) itself and maya-vision keeps only the image chunks, so mtmd adds no image begin /
end text here, a text part that is empty needs no vocabulary, and the decoder's positions are the plain ones.
Every edit is anchored on the pinned source and idempotent: a second run changes nothing, and an anchor that is
missing (another llama.cpp) stops with the file it was looking in.

    python inkling2v_patch.py <llama.cpp dir>
"""
from __future__ import annotations

import pathlib
import shutil
import sys

MARK = "inkling2v"


def edit(path: pathlib.Path, pairs: list[tuple[str, str]], count: int | None = 1):
    s = path.read_text(encoding="utf-8")
    for old, new in pairs:
        if new in s:
            continue                                   # already applied
        n = s.count(old)
        if n == 0 or (count is not None and n != count):
            raise SystemExit(f"inkling2v patch: {path}: anchor found {n} times:\n{old}")
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8", newline="\n")


def main():
    root = pathlib.Path(sys.argv[1])
    m = root / "tools" / "mtmd"
    here = pathlib.Path(__file__).resolve().parent

    edit(m / "clip-impl.h", [
        ('#define TN_CLASS_EMBD      "v.class_embd"\n',
         '#define TN_CLASS_EMBD      "v.class_embd"\n#define TN_CLASS_POS_EMBD  "v.class_position_embd.weight"\n'),
        ("    PROJECTOR_TYPE_MUSE_GLIMMER,\n", "    PROJECTOR_TYPE_MUSE_GLIMMER,\n    PROJECTOR_TYPE_INKLING2V,\n"),
        ('    { PROJECTOR_TYPE_MUSE_GLIMMER,      "muse-glimmer"},\n',
         '    { PROJECTOR_TYPE_MUSE_GLIMMER,      "muse-glimmer"},\n    { PROJECTOR_TYPE_INKLING2V,         "inkling2v"},\n'),
    ])
    edit(m / "clip-model.h", [
        ("    ggml_tensor * class_embedding = nullptr;\n",
         "    ggml_tensor * class_embedding = nullptr;\n    ggml_tensor * class_position_embedding = nullptr;\n"),
    ])
    edit(m / "models" / "models.h", [
        ("struct clip_graph_muse_glimmer : clip_graph {\n"
         "    clip_graph_muse_glimmer(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n",
         "struct clip_graph_muse_glimmer : clip_graph {\n"
         "    clip_graph_muse_glimmer(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n\n"
         "struct clip_graph_inkling2v : clip_graph {\n"
         "    clip_graph_inkling2v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}\n"
         "    ggml_cgraph * build() override;\n"
         "};\n"),
    ])
    edit(m / "CMakeLists.txt", [
        ("            models/muse-glimmer.cpp\n", "            models/muse-glimmer.cpp\n            models/inkling2v.cpp\n"),
    ])
    shutil.copy2(here / "inkling2v" / "inkling2v.cpp", m / "models" / "inkling2v.cpp")

    edit(m / "clip.cpp", [
        # the graph
        ("        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_muse_glimmer>(ctx, img);\n"
         "            } break;\n",
         "        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_muse_glimmer>(ctx, img);\n"
         "            } break;\n"
         "        case PROJECTOR_TYPE_INKLING2V:\n"
         "            {\n"
         "                builder = std::make_unique<clip_graph_inkling2v>(ctx, img);\n"
         "            } break;\n"),
        # hyperparameters
        ("                case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "                    {\n"
         "                        hparams.n_merge = 2; // pixel-shuffle downsample after the ViT\n",
         "                case PROJECTOR_TYPE_INKLING2V:\n"
         "                    {\n"
         "                        hparams.ffn_op = FFN_GELU_ERF;\n"
         "                        log_ffn_op = \"gelu_erf\";\n"
         "                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge);\n"
         "                        hparams.rope_theta = 10000.0f;\n"
         "                        hparams.set_limit_image_tokens(1, 4096);\n"
         "                        hparams.set_warmup_n_tokens(32*32); // avoid OOM on warmup\n"
         "                    } break;\n"
         "                case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "                    {\n"
         "                        hparams.n_merge = 2; // pixel-shuffle downsample after the ViT\n"),
        # tensors (class_embd, position_embd and pre_ln come from the generic path, where they are optional)
        ("            case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "                {\n"
         "                    // 3-linear MLP",
         "            case PROJECTOR_TYPE_INKLING2V:\n"
         "                {\n"
         "                    if (!model.class_embedding || !model.position_embeddings || !model.pre_ln_w || !model.pre_ln_b) {\n"
         "                        throw std::runtime_error(string_format(\"%s: missing class_embd, position_embd or pre_ln\\n\", __func__));\n"
         "                    }\n"
         "                    model.class_position_embedding = get_tensor(TN_CLASS_POS_EMBD);\n"
         "                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, \"weight\"));\n"
         "                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, \"bias\"), false);\n"
         "                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, \"weight\"));\n"
         "                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, \"bias\"), false);\n"
         "                } break;\n"
         "            case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "                {\n"
         "                    // 3-linear MLP"),
        # output grid width and height
        ("        case PROJECTOR_TYPE_STEP3VL:\n"
         "            return img->nx() / (params.patch_size * params.n_merge);\n",
         "        case PROJECTOR_TYPE_STEP3VL:\n"
         "        case PROJECTOR_TYPE_INKLING2V:\n"
         "            return img->nx() / (params.patch_size * params.n_merge);\n"),
        ("        case PROJECTOR_TYPE_STEP3VL:\n"
         "            return img->ny() / (params.patch_size * params.n_merge);\n",
         "        case PROJECTOR_TYPE_STEP3VL:\n"
         "        case PROJECTOR_TYPE_INKLING2V:\n"
         "            return img->ny() / (params.patch_size * params.n_merge);\n"),
        # the number of image tokens
        ("        case PROJECTOR_TYPE_STEP3VL:\n"
         "            {\n"
         "                int x_patch = img->nx() / (params.patch_size * params.n_merge);\n",
         "        case PROJECTOR_TYPE_STEP3VL:\n"
         "        case PROJECTOR_TYPE_INKLING2V:\n"
         "            {\n"
         "                int x_patch = img->nx() / (params.patch_size * params.n_merge);\n"),
        # 2D positions: the CLS token at 0, the patches 1-indexed
        ("    // set input per projector\n"
         "    switch (ctx->model.proj_type) {\n",
         "    // set input per projector\n"
         "    switch (ctx->model.proj_type) {\n"
         "        case PROJECTOR_TYPE_INKLING2V:\n"
         "            {\n"
         "                // CLS is at position 0, the patches are 1-indexed\n"
         "                std::vector<int32_t> positions_w{0}, positions_h{0};\n"
         "                for (int y = 0; y < pos_h; ++y) {\n"
         "                    for (int x = 0; x < pos_w; ++x) {\n"
         "                        positions_w.push_back(x + 1);\n"
         "                        positions_h.push_back(y + 1);\n"
         "                    }\n"
         "                }\n"
         "                set_input_i32(\"pos_w\", positions_w);\n"
         "                set_input_i32(\"pos_h\", positions_h);\n"
         "            } break;\n"),
        # the embedding's width
        ("        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "            return ctx->model.mm_2_w->ne[1];\n",
         "        case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "        case PROJECTOR_TYPE_INKLING2V:\n"
         "            return ctx->model.mm_2_w->ne[1];\n"),
    ])

    edit(m / "mtmd-image.h", [
        ("struct mtmd_image_preprocessor_muse_glimmer : mtmd_image_preprocessor {\n"
         "    mtmd_image_preprocessor_muse_glimmer(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}\n"
         "    mtmd_image_preproc_out preprocess(const clip_image_u8 & img) const override;\n"
         "};\n",
         "struct mtmd_image_preprocessor_muse_glimmer : mtmd_image_preprocessor {\n"
         "    mtmd_image_preprocessor_muse_glimmer(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}\n"
         "    mtmd_image_preproc_out preprocess(const clip_image_u8 & img) const override;\n"
         "};\n\n"
         "// pad bottom-right to a multiple of patch_size*n_merge; resize only if the image is above image_max_pixels\n"
         "struct mtmd_image_preprocessor_inkling2v : mtmd_image_preprocessor {\n"
         "    mtmd_image_preprocessor_inkling2v(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}\n"
         "    mtmd_image_preproc_out preprocess(const clip_image_u8 & img) const override;\n"
         "};\n"),
    ])
    edit(m / "mtmd-image.cpp", [
        ("mtmd_image_preproc_out mtmd_image_preprocessor_muse_glimmer::preprocess(const clip_image_u8 & img) const {\n",
         "mtmd_image_preproc_out mtmd_image_preprocessor_inkling2v::preprocess(const clip_image_u8 & input) const {\n"
         "    const int align = hparams.patch_size * hparams.n_merge;\n"
         "    auto size = input.get_size();\n"
         "    clip_image_u8 resized;\n"
         "    const bool too_large = hparams.image_max_pixels > 0 &&\n"
         "        int64_t(CLIP_ALIGN(size.width, align)) * CLIP_ALIGN(size.height, align) > hparams.image_max_pixels;\n"
         "    if (too_large) {\n"
         "        // the reference does not resize; this only bounds the number of image tokens\n"
         "        img_tool::calc_size_opt opts;\n"
         "        opts.align_size = align;\n"
         "        opts.max_pixels = hparams.image_max_pixels;\n"
         "        size = img_tool::calc_size_preserved_ratio(size, opts);\n"
         "        img_tool::resize(input, resized, size, hparams.image_resize_algo, PAD_NONE);\n"
         "    }\n"
         "    const clip_image_u8 & img = too_large ? resized : input;\n"
         "    const int width  = CLIP_ALIGN(size.width,  align);\n"
         "    const int height = CLIP_ALIGN(size.height, align);\n"
         "    clip_image_f32 padded;\n"
         "    padded.set_size({width, height}, img.is_placeholder(), false);\n"
         "    if (!img.is_placeholder()) {\n"
         "        std::vector<float> pixels(size_t(width) * height * 3, -1.0f);\n"
         "        const auto & src = img.get_ro_buf();\n"
         "        for (int y = 0; y < size.height; ++y) {\n"
         "            std::copy_n(src.begin() + size_t(y) * size.width * 3, size_t(size.width) * 3, pixels.begin() + size_t(y) * width * 3);\n"
         "        }\n"
         "        padded.cpy_buf(pixels);\n"
         "        // Match the reference's fused rescale and normalization, including negative padding.\n"
         "        float mean[3], std[3];\n"
         "        for (int c = 0; c < 3; ++c) {\n"
         "            mean[c] = hparams.image_mean[c] * 255.0f;\n"
         "            std[c] = hparams.image_std[c] * 255.0f;\n"
         "        }\n"
         "        padded.normalize(mean, std);\n"
         "    }\n"
         "    mtmd_image_preproc_out result;\n"
         "    result.append(hparams, padded, false);\n"
         "    return result;\n"
         "}\n\n"
         "mtmd_image_preproc_out mtmd_image_preprocessor_muse_glimmer::preprocess(const clip_image_u8 & img) const {\n"),
    ])
    edit(m / "mtmd.cpp", [
        # (maya-vision: the server writes <|content_image|><|unused_200054|> and keeps only the image chunks)
        ("            case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "                {\n"
         "                    img_beg = \"<|image_start|>\";\n",
         "            case PROJECTOR_TYPE_INKLING2V:\n"
         "                {\n"
         "                    // the chat template gives the image its own message block: <|content_image|> ... (maya-vision:\n"
         "                    // the server writes the markers, so none here - no vocabulary is needed for them)\n"
         "                    img_beg = \"\";\n"
         "                    image_preproc = std::make_unique<mtmd_image_preprocessor_inkling2v>(ctx_v);\n"
         "                } break;\n"
         "            case PROJECTOR_TYPE_MUSE_GLIMMER:\n"
         "                {\n"
         "                    img_beg = \"<|image_start|>\";\n"),
        # without a text model (maya-vision for Inkling 2): plain decoder positions, and an empty text part adds
        # nothing (with a vocabulary it tokenizes to nothing as well)
        ("    mtmd_pos_type pos_type;\n",
         "    mtmd_pos_type pos_type = MTMD_POS_TYPE_NORMAL; // (inkling2v: set by the text model when there is one)\n"),
        ("    void add_text(const std::string & txt, bool parse_special) {\n"
         "        if (vocab == nullptr) {\n",
         "    void add_text(const std::string & txt, bool parse_special) {\n"
         "        if (vocab == nullptr && txt.empty()) {\n"
         "            return; // (inkling2v: no text model, no text)\n"
         "        }\n"
         "        if (vocab == nullptr) {\n"),
    ])
    print(f"llama.cpp at {root}: {MARK} vision added")


if __name__ == "__main__":
    main()
