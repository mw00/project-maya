#include "../src/ds4/ds4_vision_weights.hpp"
#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <map>

struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t bytes;
    std::vector<uint8_t> data;
};
struct Pack {
    std::map<std::string, Tensor> tensors;
    const Tensor* get(const std::string& n) const {
        auto i = tensors.find(n);
        return i == tensors.end() ? nullptr : &i->second;
    }
    bool read(const Tensor& t, void* dst) const {
        if (t.data.size() != t.bytes) return false;
        std::memcpy(dst, t.data.data(), t.bytes);
        return true;
    }
    template <class T>
    void add(const std::string& name, const char* dtype, std::vector<int64_t> shape, const std::vector<T>& values) {
        std::vector<uint8_t> raw(values.size() * sizeof(T));
        std::memcpy(raw.data(), values.data(), raw.size());
        tensors[name] = {dtype, shape, raw.size(), raw};
    }
};

int main() {
    Pack pk;
    std::string err;
    std::vector<int64_t> shape;
    uint16_t out[64] = {};
    auto read = [&](const char* n) { return strata::ds4::vision_read_bf16(pk, n, shape, out, err); };
    pk.add<uint16_t>("bf16", "BF16", {3}, {0x3f80, 0xc000, 0x0001});
    assert(read("bf16") && out[0] == 0x3f80 && out[1] == 0xc000 && out[2] == 1);
    pk.add<float>("f32", "F32", {4}, {1.0f, -2.0f, 1.00390625f, 1.01171875f});
    assert(read("f32") && out[0] == 0x3f80 && out[1] == 0xc000 && out[2] == 0x3f80 && out[3] == 0x3f82);
    pk.add<uint16_t>("f16", "F16", {3}, {0x3c00, 0xc000, 0x0001});
    assert(read("f16") && out[0] == 0x3f80 && out[1] == 0xc000 && out[2] == 0x3380);
    std::vector<uint8_t> q(34, 2);
    q[0] = 0; q[1] = 0x38; q[2] = 0xfe; // F16 scale 0.5, then -2, +2, ...
    pk.add<uint8_t>("q8", "Q8_0", {1, 32}, q);
    assert(read("q8") && out[0] == 0xbf80 && out[1] == 0x3f80);
    pk.add<uint16_t>("w1.part.0", "BF16", {1, 2}, {0x3f80, 0x4000});
    pk.add<float>("w1.part.1", "F32", {1, 2}, {-1.0f, -2.0f});
    assert(read("w1") && shape == std::vector<int64_t>({2, 2}) &&
           out[0] == 0x3f80 && out[1] == 0x4000 && out[2] == 0xbf80 && out[3] == 0xc000);
    assert(strata::ds4::vision_read_bf16(pk, "w1", shape, nullptr, err));
    pk.tensors["w1.part.1"].shape = {2, 1};
    assert(!read("w1"));
    pk.tensors["q8"].bytes = 33;
    assert(!read("q8"));
    assert(!read("missing"));
    std::puts("vision BF16/F32/F16/Q8_0 and split gate/up: OK");
}
