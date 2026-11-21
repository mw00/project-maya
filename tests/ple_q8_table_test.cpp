// Synthetic, CPU-only GGUF table test: no model or GPU needed.
#include "strata/kernels/ngram.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void le(std::ofstream& f, uint64_t x, int n) {
    for (int i = 0; i < n; ++i) f.put((char) ((x >> (8 * i)) & 0xff));
}
void make_file(const std::filesystem::path& p, bool q8, bool truncated = false,
               uint64_t rows = 3, uint64_t offset = 0) {
    std::ofstream f(p, std::ios::binary);
    require((bool) f, "create fixture");
    f.write("GGUF", 4); le(f, 3, 4); le(f, 1, 8); le(f, 0, 8);
    const std::string name = "per_layer_token_embd.weight";
    le(f, name.size(), 8); f.write(name.data(), (std::streamsize) name.size());
    le(f, 2, 4); le(f, PLE_HEAD_DIM, 8); le(f, rows, 8);
    le(f, q8 ? 8 : 20, 4); le(f, offset, 8);  // Q8_0 or IQ4_NL, data offset
    while ((uint64_t) f.tellp() % 32) f.put(0);
    for (int row = 0; row < 3; ++row) for (int block = 0; block < 5; ++block) {
        const uint16_t scales[] = {0x3c00, 0x3800, 0x4000, 0x3400, 0xbc00};
        le(f, q8 ? scales[block] : 0x3c00, 2); // 1, 0.5, 2, 0.25, -1
        if (q8) for (int j = 0; j < 32; ++j) f.put((char) (int8_t) (row * 17 + block * 3 + j - 60));
        else for (int j = 0; j < 16; ++j) f.put(0); // IQ4_NL code 0 = -127
    }
    f.close();
    if (truncated) std::filesystem::resize_file(p, std::filesystem::file_size(p) - 1);
}
float expected(int row, int col) {
    const float scales[] = {1.0f, 0.5f, 2.0f, 0.25f, -1.0f};
    return scales[col / 32] * (float) (row * 17 + (col / 32) * 3 + col % 32 - 60);
}
void check_row(PleTable& t, unsigned row) {
    float out[PLE_HEAD_DIM]; t.read_row(row, out);
    for (int i = 0; i < PLE_HEAD_DIM; ++i)
        require(out[i] == (row < 3 ? expected((int) row, i) : 0.0f), "Q8 row mismatch");
}
}

int main(int argc, char** argv) {
    // Optional real-file probe: 1,027 deterministic rows as raw floats for an independent decoder.
    if (argc == 2) {
        PleTable t; std::string err; PleIoOptions mapped; mapped.mode = PleIo::Mmap;
        if (!t.open(argv[1], err, mapped) || t.rows() < 4) { std::cerr << err << '\n'; return 1; }
        for (uint64_t i = 0; i < 1027; ++i) {
            const uint32_t row = i < 3 ? (uint32_t) (i == 2 ? t.rows() - 1 : i)
                                       : (uint32_t) ((i * 2654435761ull + 13) % t.rows());
            float out[PLE_HEAD_DIM]; t.read_row(row, out);
            std::cout.write(reinterpret_cast<const char*>(out), sizeof out);
        }
        return std::cout ? 0 : 1;
    }
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() / ("maya-ple-q8-" + std::to_string(id));
    try {
        std::filesystem::create_directories(dir);
        const auto q8 = dir / "q8.gguf", iq4 = dir / "iq4.gguf", bad = dir / "bad.gguf";
        const auto overflow = dir / "overflow.gguf", bad_offset = dir / "offset.gguf";
        make_file(q8, true); make_file(iq4, false); make_file(bad, true, true);
        make_file(overflow, true, false, UINT64_MAX); make_file(bad_offset, true, false, 3, 32);
        PleTable t; std::string err; PleIoOptions mapped; mapped.mode = PleIo::Mmap;
        if (!t.open(q8.string(), err, mapped)) throw std::runtime_error("Q8 open failed: " + err);
        require(t.rows() == 3, "wrong row count");
        check_row(t, 0); check_row(t, 1); check_row(t, 2); check_row(t, 3);
        uint32_t rows[PLE_N_HEADS]; float output[PLE_N_HEADS * PLE_HEAD_DIM];
        for (int h = 0; h < PLE_N_HEADS; ++h) rows[h] = h % 3;
        require(t.issue(rows) && t.collect(output, err), "issue/collect failed");
        for (int h = 0; h < PLE_N_HEADS; ++h) for (int i = 0; i < PLE_HEAD_DIM; ++i)
            require(output[h * PLE_HEAD_DIM + i] == expected(rows[h], i), "issue/collect mismatch");
        t.gather(rows, output);
        for (int h = 0; h < PLE_N_HEADS; ++h) for (int i = 0; i < PLE_HEAD_DIM; ++i)
            require(output[h * PLE_HEAD_DIM + i] == expected(rows[h], i), "gather mismatch");
        require(t.gather_batch(rows, 1, output, err), "batch failed");
        for (int h = 0; h < PLE_N_HEADS; ++h) for (int i = 0; i < PLE_HEAD_DIM; ++i)
            require(output[h * PLE_HEAD_DIM + i] == expected(rows[h], i), "batch mismatch");
        uint32_t batch_rows[2 * PLE_N_HEADS]; float batch[2 * PLE_N_HEADS * PLE_HEAD_DIM];
        for (int h = 0; h < 2 * PLE_N_HEADS; ++h) batch_rows[h] = (h + 1) % 3;
        require(t.gather_batch(batch_rows, 2, batch, err), "two-token batch failed");
        for (int h = 0; h < 2 * PLE_N_HEADS; ++h) for (int i = 0; i < PLE_HEAD_DIM; ++i)
            require(batch[h * PLE_HEAD_DIM + i] == expected(batch_rows[h], i), "two-token batch mismatch");
        PleIoOptions io; io.mode = PleIo::Direct;
        require(t.open(q8.string(), err, mapped) && t.is_open(), "reopen failed");
        require(!t.open(q8.string(), err, io) && err.find("mmap") != std::string::npos, "Q8 direct must fail");
        require(!t.open(bad.string(), err, mapped), "truncated Q8 must fail");
        require(!t.open(overflow.string(), err, mapped), "overflow Q8 must fail");
        require(!t.open(bad_offset.string(), err, mapped), "out-of-bounds Q8 must fail");
        require(t.open(iq4.string(), err, mapped), "IQ4 reopen failed");
        float iq[PLE_HEAD_DIM]; t.read_row(1, iq);
        for (float x : iq) require(x == -127.0f, "IQ4 behavior changed");
        t.close();
        require(t.open(q8.string(), err, mapped), "Q8 reopen after IQ4 failed");
        check_row(t, 2); t.close();
        std::filesystem::remove_all(dir);
        std::cout << "Q8 and IQ4 mmap table checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(dir);
        std::cerr << e.what() << '\n'; return 1;
    }
}
