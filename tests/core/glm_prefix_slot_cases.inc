  else if (which.rfind("branch-", 0) == 0) {
    if (!branch_enabled) fail("candidate did not enable in fixture");
    auto a = std::vector<int32_t>(4000, 17), b = a, c = a;
    b[1883] = 19; c[1883] = 23;
    auto seed = [&] { model.live_pos = 1883; branch_save(b, 1883); };
    auto count_files = [&] {
      size_t n = 0; std::error_code ec;
      for (const auto& e : std::filesystem::directory_iterator(slot_dir, ec)) ++n;
      return n;
    };
    if (which == "branch-ready-write" || which == "branch-ready-publish" || which == "branch-ready-serialize") {
      seed();
      if (slots.size()!=1 || slots[0].data) fail("disk seed missing");
      const auto old_path = slots[0].path;
      const auto old_tokens = slots[0].tokens;
      snap_tokens.assign(3000, 29); snap_in_slot = false;
      if (which != "branch-ready-publish") model.fail_file = true;
      else model.fail_metadata = true;
      const auto writes = model.file_saves;
      slot_put("");
      if (which == "branch-ready-write" ? model.file_saves != writes : model.file_saves != writes+1)
        fail("ordinary failure test did not reach the intended capacity/serialization/publication phase");
      if (snap_in_slot || slots.size()!=1 || slots[0].path!=old_path || slots[0].tokens!=old_tokens ||
          !std::filesystem::exists(old_path) || slot_take(c, 0, true)!=1883)
        fail("failed ordinary snapshot destroyed the previous usable disk slot");
    } else if (which == "branch-ready-seed") {
      seed(); if (!slot_write_file(0)) fail("cold seed spill failed");
      b[0] = c[0] = 31; seed(); if (!slot_write_file(1)) fail("hot seed spill failed");
      const auto now = std::filesystem::file_time_type::clock::now();
      std::filesystem::last_write_time(meta_of(slots[0].path), now - std::chrono::hours(1));
      std::filesystem::last_write_time(meta_of(slots[1].path), now);
    } else if (which == "branch-ready-reindex") {
      c[0] = 31;
      if (slots.size()!=1 || slot_take(c, 0, true)!=1883) fail("reduced restart quota discarded hottest state");
    } else if (which == "branch-ready-short" || which == "branch-ready-short-disk" || which == "branch-ready-short-reindex") {
      auto short_key = std::vector<int32_t>(12, 41);
      if (which != "branch-ready-short-reindex") {
        const auto prompt = std::vector<int32_t>(10, 41);
        const auto emitted = std::vector<int32_t>(2, 41);
        model.live_pos=12;
        completion_save(prompt, emitted, "stop", false);
      }
      short_key.push_back(42);
      if (slots.size()!=1 || slot_take(short_key,0,true)!=12) fail("short-key RAM/disk/restart semantics disagree");
    } else if (which == "branch-learn") {
      if (prefix_admission.observe(a, 0) != 0 || slot_take(a, 0, true) != 0) fail("first request falsely hit");
      const size_t p = prefix_admission.observe(b, 0);
      if (p != 1883 || slot_take(b, 0, true) != 0) fail("metadata-only second request falsely hit");
      const auto old_pos = model.snap_pos; const auto old_state = model.snap_state;
      model.live_pos = p; model.context = 1; branch_save(b, p);
      if (slots.size() != 1 || !slots[0].data || slots[0].branch_id == 0)
        fail("learned RAM state identity absent");
      if (model.snap_pos != old_pos || model.snap_state != old_state || model.snapshot_saves != 0)
        fail("live boundary save overwrote old prompt snapshot");
      model.context = 2; model.live_pos = 999;
      if (slot_take(c, 0, true) != 1883 || model.loaded_origin != 1 || model.context != 2 || model.live_pos != 1883)
        fail("third request did not restore saved source state into destination context");
      if (slot_take(c, 1883, true) != 0) fail("equal-length floor should keep existing snapshot");
    } else if (which == "branch-position") {
      model.live_pos = 1884; branch_save(b, 1883);
      if (!slots.empty() || model.live_saves != 0) fail("uncomputed or wrong-position state was saved");
    } else if (which == "branch-save-fail") {
      const auto old = model.snap_state; model.fail_live = true; seed();
      if (!slots.empty() || model.snap_state != old || model.snapshot_saves != 0) fail("failed save published or destroyed old snapshot");
    } else if (which == "branch-direct-disk" || which == "branch-flush") {
      seed();
      if (slots.size() != 1 || slots[0].branch_id == 0) fail("no learned state to spill");
      if (which == "branch-direct-disk") {
        if (slots[0].data || slots[0].path.empty()) fail("RAM budget fallback did not stream to disk");
      } else {
        if (!slots[0].data || !slot_write_file(0) || slots[0].data) fail("RAM learned state did not spill");
      }
      if (slots[0].path.find("branch-") == std::string::npos || !std::filesystem::exists(meta_of(slots[0].path)))
        fail("learned disk classification missing");
      if (slot_take(c, 0, true) != 1883) fail("learned disk state did not restore");
    } else if (which == "branch-reindex") {
      if (slots.size() != 1 || slots[0].branch_id == 0 || slots[0].data || slots[0].prefix)
        fail("restart lost learned classification");
      if (slot_take(c, 0, true) != 1883) fail("restart could not restore exact state");
      slot_drop(0);
      if (!slots.empty() || slot_take(c, 0, true) != 0) fail("deleted state remained hittable");
    } else if (which == "branch-corrupt") {
      seed(); if (slots.size() != 1 || !slot_write_file(0)) fail("corrupt fixture seed failed");
      const auto path = slots[0].path;
      std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary); f.put('X'); f.close();
      const size_t reused = slot_take(c, 0, true);
      std::fprintf(stderr, "corrupt check: reused=%zu slots=%zu file_exists=%d model_loads=%d\n",
                   reused, slots.size(), int(std::filesystem::exists(path)), model.loads);
      if (reused != 0 || !slots.empty() || std::filesystem::exists(path) || model.loads != 0)
        fail("corrupt learned payload reached model or retained stale reference");
    } else if (which == "branch-supersede") {
      seed(); snap_tokens = std::vector<int32_t>(b.begin(), b.begin() + 3000); snap_in_slot = false;
      slot_put("");
      if (std::count_if(slots.begin(), slots.end(), [](const auto& s) { return s.branch_id != 0; }) != 1)
        fail("ordinary longer session superseded learned prefix");
    } else if (which == "branch-low-memory") {
      fixture_available = uint64_t{32} << 30; seed();
      if (!slots.empty() || model.live_saves || model.vector_saves || model.file_saves || count_files())
        fail("low memory allocated/serialized/published before guard");
    } else if (which == "branch-peak-disk") {
      const uint64_t temporary = Glm5Model::slot_live_copy_bytes + 2 * 1883 * sizeof(int32_t) + (1u << 20);
      fixture_available = (uint64_t{32} << 30) + temporary + 4; // fits stream peak, not stream+state(8)
      seed();
      if (slots.size() != 1 || slots[0].data || model.live_saves != 1 || model.vector_saves)
        fail("RAM temporary peak did not select streamed disk fallback");
    } else if (which == "branch-peak-reject") {
      fixture_available = (uint64_t{32} << 30) + Glm5Model::slot_live_copy_bytes;
      seed(); if (!slots.empty() || model.live_saves || count_files()) fail("stream temporary peak was not protected");
    } else if (which == "branch-load-memory") {
      seed(); if (slots.size() != 1 || !slot_write_file(0)) fail("load memory fixture seed failed");
      fixture_available = (uint64_t{32} << 30) + (32u << 20);
      if (slot_take(c, 0, true) || model.loads || slots.size() != 1) fail("disk restore did not guard 64MiB temp before loading");
    } else if (which == "branch-disk-floor") {
      fixture_disk_available = uint64_t{8} << 30; seed();
      if (!slots.empty() || model.live_saves || count_files()) fail("disk floor did not include new payload+metadata");
    } else if (which == "branch-write-fail") {
      const auto bad = root / "bad-directory"; std::ofstream(bad).put('x'); slot_dir = bad;
      seed(); if (!slots.empty()) fail("unwritable disk published state");
    } else if (which == "branch-atomic") {
      seed(); if (slots.size() != 1 || !slots[0].data) fail("atomic fixture seed failed");
      const auto saved = slots[0].data;
      const auto bytes = *saved;
      if (publish_files(slots[0], [&](const std::string& tmp) {
            std::ofstream f(tmp, std::ios::binary); f.write("1234", 4); return false;
          }) || slots[0].data != saved || *slots[0].data != bytes || !slots[0].path.empty() || count_files())
        fail("failed partial write destroyed old RAM state or left publish residue");
      if (publish_files(slots[0], [&](const std::string& tmp) {
            std::ofstream f(tmp, std::ios::binary); f.write(reinterpret_cast<const char*>(saved->data()), saved->size()); f.close();
            std::filesystem::create_directory(meta_of(tmp.substr(0, tmp.size() - 4)) + ".tmp");
            return true;
          }) || slots[0].data != saved || !slots[0].path.empty() || count_files())
        fail("metadata publish failure destroyed old state or left residue");
    } else if (which == "branch-budget64") {
      if (slot_disk_budget != (uint64_t{64} << 30)) fail("candidate default disk budget is not 64GiB");
      Slot synthetic{{1}, "", 0, 1};
      const uint64_t metadata = slot_identity.size() + 48 + sizeof(int32_t);
      synthetic.bytes = slot_disk_budget - metadata;
      if (!disk_fits(synthetic)) fail("exact64GiB including metadata rejected");
      ++synthetic.bytes;
      if (disk_fits(synthetic)) fail("64GiB plus one byte admitted");
    } else if (which == "branch-budget-cap") {
      if (slot_disk_budget != (uint64_t{64} << 30)) fail("configured budget above64GiB not clamped");
    } else if (which == "branch-zero-budget") {
      seed(); if (slot_disk_budget || !slots.empty() || model.live_saves) fail("zero disk budget attempted admission");
    } else if (which == "branch-count") {
      model.live_pos = 1883;
      for (int i = 0; i < 17; ++i) { b[0] = i; branch_save(b, 1883); }
      if (slots.size() != 16 || model.live_saves != 17 || learned_max != 16) fail("full learned catalogue did not replace its cold entry");
      b[0] = 16; if (slot_take(b, 0, true) != 1883) fail("new seventeenth branch not restorable");
      b[0] = 0; if (slot_take(b, 0, true) != 0) fail("retired first branch still hittable");
    } else if (which.rfind("branch-lru-", 0) == 0) {
      auto family = [&](int32_t id) { auto p = b; p[0] = id; return p; };
      auto put_family = [&](int32_t id) { model.live_pos = 1883; branch_save(family(id), 1883); };
      auto find_family = [&](int32_t id) {
        return std::find_if(slots.begin(), slots.end(), [&](const Slot& s) { return !s.tokens.empty() && s.tokens[0] == id; });
      };
      auto require_old_pair = [&] {
        if (slots.size() != 2 || find_family(1)==slots.end() || find_family(2)==slots.end() || find_family(3)!=slots.end())
          fail("failed replacement changed old catalogue or published new state");
        for (const auto& s : slots) if (!s.data && (!std::filesystem::exists(s.path) || !std::filesystem::exists(meta_of(s.path))))
          fail("failed replacement removed old payload or metadata");
      };
      if (which == "branch-lru-reindex") {
        if (slots.size()!=2 || find_family(1)==slots.end() || find_family(2)!=slots.end() || find_family(3)==slots.end() || count_files()!=4)
          fail("restart recovered retired branch or lost replacement");
        for (const auto& s : slots) if (!s.branch_id || s.data) fail("restart lost learned disk classification");
        if (slot_take(family(3),0,true)!=1883 || slot_take(family(2),0,true)!=0) fail("restart replacement lookup incorrect");
      } else if (which == "branch-lru-tokens") {
        slot_flush_on_exit.enabled = false; // Metadata-budget fixture stays in RAM; no 32MiB sidecar flush.
        for (int32_t id = 1; id <= 8; ++id) {
          Slot s{std::vector<int32_t>(branch::max_path_tokens, id), "", 8, ++slot_clock};
          s.branch_id = ++branch_seq; s.data = ram(); slots.push_back(std::move(s));
        }
        slot_touch(slots[0]);
        put_family(9);
        size_t total_tokens = 0; for (const auto& s : slots) total_tokens += s.tokens.size();
        if (slots.size()!=8 || find_family(1)==slots.end() || find_family(2)!=slots.end() || find_family(9)==slots.end() ||
            total_tokens > branch::retained_token_budget || model.live_saves != 1)
          fail("token metadata budget did not retire only the cold learned entry");
        if (slot_take(family(9),0,true)!=1883) fail("token-budget replacement cannot restore");
      } else {
        put_family(1); put_family(2);
        if (slots.size()!=2) fail("LRU fixture could not fill two learned entries");
        slot_touch(*find_family(1)); // recency only: do not accidentally protect it as the active snapshot
        const auto hot_id = find_family(1)->branch_id, cold_id = find_family(2)->branch_id;
        const auto cold_path = find_family(2)->path, hot_path = find_family(1)->path;
        const auto cold_data = find_family(2)->data, hot_data = find_family(1)->data;
        if (which=="branch-lru-bound") {
          snap_tokens = find_family(1)->tokens;
          const auto other_tokens = find_family(2)->tokens;
          branch_bound_elsewhere = [other_tokens](const auto& t) { return t == other_tokens; };
          Slot ordinary{{99}, "", 8, 0}; ordinary.data=ram(); slots.push_back(std::move(ordinary));
          put_family(3);
          if (slots.size()!=3 || model.live_saves!=2 || find_family(3)!=slots.end()) fail("all-bound learned states were evicted");
          branch_bound_elsewhere = {};
          put_family(3);
          if (slots.size()!=3 || find_family(1)==slots.end() || find_family(2)!=slots.end() || find_family(3)==slots.end() ||
              find_family(99)==slots.end() || find_family(99)->branch_id) fail("replacement touched bound state or ordinary slot");
        } else if (which=="branch-lru-save-fail" || which=="branch-lru-meta-fail") {
          model.fail_live = which=="branch-lru-save-fail";
          model.fail_metadata = which=="branch-lru-meta-fail";
          put_family(3); require_old_pair();
          if (find_family(1)->branch_id!=hot_id || find_family(2)->branch_id!=cold_id ||
              find_family(1)->data!=hot_data || find_family(2)->data!=cold_data)
            fail("failed replacement changed original state identity or buffers");
          if (which=="branch-lru-meta-fail" && count_files()!=4) fail("metadata failure left partial replacement files");
          if (slot_take(family(1),0,true)!=1883 || slot_take(family(2),0,true)!=1883 || slot_take(family(3),0,true)!=0)
            fail("failure recovery did not preserve old states and reject the new state");
        } else if (which=="branch-lru-oversized") {
          model.declared_state_bytes=24; put_family(3); require_old_pair();
          if (model.live_saves!=2) fail("individually oversized state reached serializer");
        } else if (which=="branch-lru-pressure") {
          fixture_available=uint64_t{32}<<30; put_family(3); require_old_pair();
          fixture_available=uint64_t{128}<<30; fixture_disk_available=uint64_t{8}<<30;
          put_family(3); require_old_pair();
          if (model.live_saves!=2 || count_files()!=4) fail("resource shortage staged or retired a state");
        } else if (which=="branch-lru-staging") {
          if (slot_disk_budget!=20000 || slot_disk_bytes()>slot_disk_budget) fail("staging fixture budget invalid");
          put_family(3); require_old_pair();
          if (model.live_saves!=2 || count_files()!=4) fail("replacement ignored old files during disk staging");
        } else {
          put_family(3);
          if (slots.size()!=2 || find_family(1)==slots.end() || find_family(2)!=slots.end() || find_family(3)==slots.end() ||
              find_family(1)->branch_id!=hot_id || find_family(3)->branch_id==cold_id || model.live_saves!=3)
            fail("LRU did not replace cold state with an independently materialized new state");
          if (which=="branch-lru-bytes" && (learned_budget!=16 || learned_max!=16)) fail("byte-limit case accidentally tested count limit");
          if (which=="branch-lru-disk" && (cold_path.empty() || std::filesystem::exists(cold_path) ||
              std::filesystem::exists(meta_of(cold_path)) || !std::filesystem::exists(hot_path) || count_files()!=4))
            fail("disk replacement left old files or removed hot state");
          if (slot_take(family(3),0,true)!=1883 || slot_take(family(2),0,true)!=0) fail("replacement lookup used retired state or missed new one");
        }
      }
    } else if (which == "branch-ordinary-low") {
      snap_tokens = a; fixture_available = uint64_t{32} << 30;
      slot_put("");
      if (snap_in_slot || !slots.empty() || model.vector_saves || model.file_saves) fail("ordinary snapshot allocated before candidate memory guard");
    } else if (which == "branch-legacy-meta") {
      snap_tokens = a; slot_put("");
      if (slots.size() != 1 || slots[0].branch_id || !slot_write_file(0)) fail("ordinary old-format write failed");
      std::vector<int32_t> tokens; uint64_t bytes = 0, img = 0;
      if (!meta_read(meta_of(slots[0].path), tokens, bytes, img) || tokens != a || bytes != 8)
        fail("old metadata no longer roundtrips while candidate enabled");
    } else if (which == "branch-legacy-reindex") {
      if (slots.size() != 1 || slots[0].branch_id || slots[0].tokens != a) fail("restart rejected compatible old metadata");
    } else return 64;
  }
