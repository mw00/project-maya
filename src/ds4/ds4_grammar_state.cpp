// Adapted from Xolotl (MIT), copyright (c) 2026 Niko1221 and the Strata contributors.
// Compile the unmodified, pinned llama.cpp sampler and expose a memoization key.
// This reads its pushdown state; parsing and every grammar decision stay in
// llama.cpp. Including its source avoids mirroring a private context layout or
// modifying the shared vendor tree. Private fields use their actual upstream
// types; maintain this key alongside the pinned grammar state definition.
#ifndef MAYA_DS4_LLAMA_SAMPLER_SOURCE
#error "q36 grammar state needs the pinned llama.cpp sampler source"
#endif
#include MAYA_DS4_LLAMA_SAMPLER_SOURCE
#include <cstdint>

bool ds4_grammar_state_key(const llama_sampler * sampler, std::string & key) {
    if (!sampler || sampler->iface != &llama_sampler_grammar_i) return false;
    const auto * ctx = static_cast<const llama_sampler_grammar *>(sampler->ctx);
    if (!ctx->grammar) return false;
    const auto & g = *ctx->grammar;
    // q36 implements its delayed activation outside the normal sampler. Fall
    // back to uncached llama.cpp if a different caller ever uses lazy triggers.
    if (g.lazy || g.awaiting_trigger || !g.trigger_tokens.empty() || !g.trigger_patterns.empty() ||
        !g.trigger_buffer.empty() || !g.trigger_buffer_positions.empty()) return false;

    using Frame = std::pair<size_t, size_t>;
    std::vector<std::vector<Frame>> stacks;
    for (const auto & stack : g.stacks) {
        std::vector<Frame> frames;
        for (const auto * frame : stack) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(frame);
            bool found = false;
            for (size_t rule = 0; rule < g.rules.size(); ++rule) {
                const auto & items = g.rules[rule];
                const uintptr_t start = reinterpret_cast<uintptr_t>(items.data());
                if (address >= start && address - start < items.size() * sizeof(llama_grammar_element) &&
                    (address - start) % sizeof(llama_grammar_element) == 0) {
                    frames.emplace_back(rule, (address - start) / sizeof(llama_grammar_element));
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        stacks.push_back(std::move(frames));
    }
    // Stacks describe a set of possible parses. Ordering and duplicate paths
    // do not change the language accepted by the native grammar.
    std::sort(stacks.begin(), stacks.end());
    stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());
    key.clear();
    const auto put = [&](uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) key.push_back(char(value >> (8 * i)));
    };
    put(static_cast<uint64_t>(static_cast<int64_t>(g.partial_utf8.n_remain)));
    // decode_utf8 starts a fresh value when no continuation bytes remain.
    put(g.partial_utf8.n_remain ? g.partial_utf8.value : 0);
    put(stacks.size());
    for (const auto & stack : stacks) {
        put(stack.size());
        for (const auto & frame : stack) { put(frame.first); put(frame.second); }
    }
    return true;
}
