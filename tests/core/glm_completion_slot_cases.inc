  else if (which.rfind("completion-", 0) == 0) {
    // These cases catch missing completion routing, an overlong token key, destructive
    // snapshot replacement, unsafe gap binding, and publication before successful save.
    const std::vector<int32_t> prompt = which == "completion-publish-fail"
        ? std::vector<int32_t>(1100, 10) : std::vector<int32_t>{10, 11, 12};
    const std::vector<int32_t> emitted{20, 21, 22};
    snap_tokens = prompt; model.snap_pos = 3; model.snap_state = 777;
    model.live_pos = prompt.size() + 2;
    struct Gen {
      int ctx = 1, rid = 42; std::vector<int32_t> prompt, emitted;
      bool keep = false, pause = false, done = false, has_images = false;
      const char* finish = "length";
    };
    struct Context { bool busy = true; uint64_t used = 0; };
    std::array<Context, 3> ctxs; uint64_t ctx_clock = 0;
    bool in_read_gap = false; int binds = 0;
    const auto bind_ctx = [&](int ctx) { ++binds; model.context = ctx; };
    const auto pause_snapshot = [&](const auto&, const auto&, bool) {
      fail("normal completion entered PAUSE path");
    };
    std::atomic<bool> pause_req{false}, pause_keep{false};
    // EXTRACTED_COMPLETION_ROUTES
    const auto check_old = [&] {
      if (snap_tokens != prompt || model.snap_pos != 3 || model.snap_state != 777 || model.snapshot_saves)
        fail("completion destroyed the valid prompt snapshot");
    };
    const auto check_saved = [&](const std::vector<int32_t>& want) {
      check_old();
      if (slots.size() != 1 || slots[0].tokens != want || !slots[0].branch_id)
        fail("normal completion did not publish the exact committed token key");
    };
    if (which == "completion-image-text" || which == "completion-text-image" ||
        which == "completion-image-text-gap" || which == "completion-text-image-gap") {
      // Admission records A's source identity. B then replaces global image data;
      // neither immediate completion nor a later gap flush may use B's identity.
      const bool image_a = which.rfind("completion-image-", 0) == 0;
      Gen g; g.prompt = prompt; g.emitted = emitted;
      if (image_a) cur_img_pos = {1}; else cur_img_pos.clear();
      capture_request_images(g);
      if (image_a) cur_img_pos.clear(); else cur_img_pos = {2};
      in_read_gap = which.ends_with("-gap");
      model.context = 2;
      finish_route(g);
      if (!g.done) fail("image identity check changed request completion");
      if (in_read_gap) {
        if (binds) fail("image identity fix rebound context during prefill gap");
        in_read_gap = false;
        flush_pauses();
      }
      if (ctxs[1].busy) fail("image identity check stranded completed owner");
      if (image_a) {
        if (!slots.empty() || model.live_saves)
          fail("image A was published under text B's no-images token identity");
      } else {
        check_saved({10, 11, 12, 20, 21});
      }
    } else if (which == "completion-off" || which == "completion-empty" || which == "completion-cancel") {
      completion_save(prompt, which == "completion-empty" ? std::vector<int32_t>{} : emitted,
                      which == "completion-cancel" ? "cancel" : "length", false);
      if (!slots.empty() || model.live_saves) fail("disabled/empty/cancel completion saved");
    } else if (which == "completion-threshold" || which == "completion-default-min") {
      completion_save(prompt, emitted, "length", false);
      if (!slots.empty() || model.live_saves) fail("short committed suffix bypassed minimum increment");
      model.live_pos = which == "completion-default-min" ? 1027 : 6;
      const auto long_emitted = which == "completion-default-min" ? std::vector<int32_t>(1024, 20) : emitted;
      completion_save(prompt, long_emitted, "length", false);
      if (slots.size() != 1 || slots[0].tokens.size() != size_t(model.live_pos))
        fail("minimum committed increment boundary was not admitted");
    } else if (which == "completion-position") {
      for (int64_t p : {-1, 0, 2, 3, 7}) {
        model.live_pos = p; completion_save(prompt, emitted, "length", false);
        if (!slots.empty() || model.live_saves) fail("invalid position saved/truncated a state");
      }
    } else if (which == "completion-shards") {
      model.shards_agree = false; completion_save(prompt, emitted, "length", false);
      if (!slots.empty()) fail("disagreeing target shards published");
    } else if (which == "completion-memory" || which == "completion-save-fail" || which == "completion-publish-fail") {
      model.live_pos = prompt.size() - 1; branch_save(prompt, prompt.size() - 1);
      if (slots.size() != 1) fail("old slot seed failed");
      const auto old_tokens = slots[0].tokens; const auto old_data = slots[0].data;
      model.live_pos = prompt.size() + 2;
      if (which == "completion-memory") fixture_available = uint64_t{32} << 30;
      if (which == "completion-save-fail") model.fail_live = true;
      if (which == "completion-publish-fail") model.fail_metadata = true;
      completion_save(prompt, emitted, "length", false);
      if (slots.size() != 1 || slots[0].tokens != old_tokens || slots[0].data != old_data)
        fail("rejected/failed completion damaged old cache");
    } else if (which == "completion-context" || which == "completion-gap") {
      Gen g; g.prompt = prompt; g.emitted = emitted;
      in_read_gap = which == "completion-gap";
      model.context = 2; finish_route(g);
      if (!g.done) fail("completion altered finish semantics");
      if (in_read_gap) {
        if (binds || !slots.empty() || !ctxs[1].busy || deferred_pauses.size() != 1)
          fail("gap completion switched prefill context or released pending owner");
        in_read_gap = false; flush_pauses();
      }
      if (binds != 1 || ctxs[1].busy || model.context != 1) fail("wrong completion context lifecycle");
      check_saved({10, 11, 12, 20, 21});
      auto next = std::vector<int32_t>{10, 11, 12, 20, 21, 22, 30};
      model.context = 2;
      if (slot_take(next, 0, true) != 5 || model.loaded_origin != 1 || model.context != 2)
        fail("completion could not restore source state in another context");
    } else if (which == "completion-single") {
      single_route(prompt, emitted, "stop"); check_saved({10, 11, 12, 20, 21});
    } else if (which == "completion-dflash") {
      // Callback-time speculative target position is out of range. Only the
      // post-decode committed boundary may be saved; draft IDs are not emitted IDs.
      model.live_pos = 9; completion_save(prompt, emitted, "stop", false);
      if (!slots.empty()) fail("unaccepted DFlash draft state was cached");
      model.live_pos = 6; single_route(prompt, emitted, "stop");
      check_saved({10, 11, 12, 20, 21, 22});
    } else {
      if (which == "completion-full") model.live_pos = 6;
      completion_save(prompt, emitted, which == "completion-stop" ? "stop" : "length", false);
      if (which == "completion-full") check_saved({10, 11, 12, 20, 21, 22});
      else check_saved({10, 11, 12, 20, 21});
    }
    // Restore itself legitimately changes the snapshot in the context case.
    if (which != "completion-context" && which != "completion-gap") check_old();
  }
