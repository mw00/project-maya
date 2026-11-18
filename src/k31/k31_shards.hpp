// src/k31/k31_shards.hpp - maya-k31: a GGUF that llama.cpp's gguf-split cut into shards, read as one file.
//
// "<name>-00001-of-00014.gguf" names the set: the first shard holds the metadata (split.count, split.no 0,
// split.tensors.count), every shard holds a part of the tensor directory and its data.  Shards gives the reader the
// GgufFile calls it uses (get, get_full, tensors, tensor_data) over the whole set: tensors() is every shard's
// directory in shard order, each entry's offset still relative to its own shard's data, and tensor_data() looks the
// shard up.  A path that is not a shard name, or a file without split keys, is a set of one.
//
// allow_missing: open the shards that are on disk and say which are not (pack-info on a set still downloading); the
// model loader always asks for all of them.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace k31 {

class Shards {
public:
    explicit Shards(const std::string& path, bool allow_missing = false) {
        auto first = std::make_unique<strata::GgufFile>(path);
        const strata::MetaValue* count = first->get("split.count");
        const strata::MetaValue* no = first->get("split.no");
        const strata::MetaValue* nt = first->get("split.tensors.count");
        n_expected_ = count ? (int) count->u : 1;
        if (n_expected_ < 1 || n_expected_ > 9999) throw std::runtime_error("split.count " + std::to_string(n_expected_));
        if (n_expected_ > 1) {
            if (!no || no->u != 0) throw std::runtime_error(path + " is not the first shard (split.no " +
                                                            (no ? std::to_string(no->u) : std::string("missing")) + ")");
            std::smatch m;
            static const std::regex kName(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");
            if (!std::regex_match(path, m, kName) || std::stoi(m[3].str()) != n_expected_)
                throw std::runtime_error(path + ": split.count is " + std::to_string(n_expected_) +
                                         " but the name is not <prefix>-00001-of-" + fmt5(n_expected_) + ".gguf");
            tensors_count_ = nt ? (long long) nt->u : -1;
            const std::string prefix = m[1].str();
            paths_.push_back(path);
            files_.push_back(std::move(first));
            for (int i = 2; i <= n_expected_; ++i) {
                const std::string p = prefix + "-" + fmt5(i) + "-of-" + fmt5(n_expected_) + ".gguf";
                if (!std::filesystem::exists(p)) {
                    if (!allow_missing) throw std::runtime_error("shard " + p + " is missing");
                    missing_.push_back(p);
                    continue;
                }
                auto f = std::make_unique<strata::GgufFile>(p);
                const strata::MetaValue* c2 = f->get("split.count");
                const strata::MetaValue* n2 = f->get("split.no");
                if (!c2 || !n2 || (int) c2->u != n_expected_ || (int) n2->u != i - 1)
                    throw std::runtime_error(p + ": split keys do not match (split.no " +
                                             (n2 ? std::to_string(n2->u) : std::string("missing")) + ", want " +
                                             std::to_string(i - 1) + ")");
                paths_.push_back(p);
                files_.push_back(std::move(f));
            }
        } else {
            paths_.push_back(path);
            files_.push_back(std::move(first));
        }
        for (size_t s = 0; s < files_.size(); ++s)
            for (const auto& t : files_[s]->tensors()) {
                tensors_.push_back(t);
                shard_.push_back((int) s);
            }
        if (missing_.empty() && tensors_count_ >= 0 && (long long) tensors_.size() != tensors_count_)
            throw std::runtime_error("the shards hold " + std::to_string(tensors_.size()) +
                                     " tensors, split.tensors.count says " + std::to_string(tensors_count_));
    }

    /// --dense-override: every tensor of `path` (a GGUF, or the first shard of a set) that this set holds under the same
    /// name and shape replaces it: its type and bytes come from that file from then on (the loader, the planner and
    /// the RAM tier's reads all see it).  The routed experts ("_exps") and unknown names are refused.  The count.
    int overlay(const std::string& path) {
        Shards o(path);
        std::map<std::string, size_t> at;
        for (size_t i = 0; i < tensors_.size(); ++i) at[tensors_[i].name] = i;
        int n = 0;
        for (size_t s = 0; s < o.files_.size(); ++s) {
            const int fi = (int) files_.size();
            bool used = false;
            for (const auto& t : o.files_[s]->tensors()) {
                const auto it = at.find(t.name);
                if (it == at.end()) throw std::runtime_error(path + ": " + t.name + " is not a tensor of the model");
                if (t.name.find("_exps") != std::string::npos)
                    throw std::runtime_error(path + ": " + t.name + " is a routed expert (an override holds dense tensors only)");
                if (t.shape != tensors_[it->second].shape) throw std::runtime_error(path + ": " + t.name + " has another shape");
                tensors_[it->second] = t;
                shard_[it->second] = fi;
                used = true;
                ++n;
            }
            if (used) {
                paths_.push_back(o.paths_[s]);
                files_.push_back(std::move(o.files_[s]));
            }
        }
        return n;
    }

    // the GgufFile calls k31::read and the loader use
    const strata::MetaValue* get(const std::string& k) const { return files_[0]->get(k); }
    bool get_full(const std::string& k, strata::MetaValue& out) const { return files_[0]->get_full(k, out); }
    const std::map<std::string, strata::MetaValue>& metadata() const { return files_[0]->metadata(); }
    uint32_t version() const { return files_[0]->version(); }
    const std::vector<strata::TensorInfo>& tensors() const { return tensors_; }
    const uint8_t* tensor_data(const strata::TensorInfo& t) const { return file_of(t).tensor_data(t); }

    // per shard
    const strata::GgufFile& file_of(const strata::TensorInfo& t) const { return *files_[(size_t) shard_of(t)]; }
    int shard_of(const strata::TensorInfo& t) const {
        const ptrdiff_t i = &t - tensors_.data();
        if (i < 0 || (size_t) i >= tensors_.size()) throw std::runtime_error("k31::Shards: not one of the set's tensors");
        return shard_[(size_t) i];
    }
    const std::string& path_of(const strata::TensorInfo& t) const { return paths_[(size_t) shard_of(t)]; }
    size_t n_open() const { return files_.size(); }
    const strata::GgufFile& file(size_t s) const { return *files_[s]; }
    const std::string& path(size_t s) const { return paths_[s]; }
    int n_expected() const { return n_expected_; }
    long long tensors_count() const { return tensors_count_; }
    const std::vector<std::string>& missing() const { return missing_; }
    bool complete() const { return missing_.empty(); }

private:
    static std::string fmt5(int i) {
        char b[16];
        std::snprintf(b, sizeof b, "%05d", i);
        return b;
    }
    std::vector<std::unique_ptr<strata::GgufFile>> files_;
    std::vector<std::string> paths_, missing_;
    std::vector<strata::TensorInfo> tensors_;
    std::vector<int> shard_;
    int n_expected_ = 1;
    long long tensors_count_ = -1;
};

}  // namespace k31
