"""Contract for genuine context-by-depth GLM target verification windows."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "include/strata/core/glm_model.hpp").read_text()
ROWS = (ROOT / "src/core/glm_rows.cu").read_text()
GENERATE = (ROOT / "src/program/generate.cpp").read_text()


class GlmMultiWindowContract(unittest.TestCase):
    def test_depth_window_api_accepts_context_slot_and_token_rows(self):
        self.assertIn("multi_window", HEADER,
                      "missing depth>1 target API; multi_rows accepts only one token per context")
        self.assertIn("commit_window_prefixes", HEADER,
                      "missing per-context accepted-prefix commit API")

    def test_rows_keep_independent_residuals_for_duplicate_context_slots(self):
        self.assertIn("row_residual", ROWS,
                      "duplicate context×depth rows need private residual scratch, not shared Rc/Ro state")
        self.assertIn("rows_bind_window", ROWS,
                      "window binding must map repeated sequence IDs to distinct token positions")

    def test_window_head_uses_each_row_final_residual(self):
        start = ROWS.index("bool Glm5Model::rows_head")
        end = ROWS.index("bool Glm5Model::multi_rows", start)
        body = ROWS[start:end]
        self.assertIn("head_prep(rw[i].Ro", body,
                      "duplicate context slots share SeqCtx.state; each target pick must read its row's residual")
        self.assertNotIn("head_prep(rw[i].state", body)

    def test_scheduler_reuses_queued_drafts_in_later_pair_windows(self):
        start = GENERATE.index("const auto multi_step = [&] {")
        end = GENERATE.index("Gen* only = gens.size()", start)
        body = GENERATE[start:end]
        self.assertRegex(
            body,
            r"if\s*\(g\.dflash_next\s*<\s*g\.dflash_drafts\.size\(\)\)\s*\{\s*window_gens\.push_back\(&g\);\s*continue;",
            "queued drafts must be included in later four-context windows instead of falling back to one row per step",
        )

    def test_32_row_depth_is_selectable_without_changing_default(self):
        self.assertIn("kMaxMvNt = 32", (ROOT / "include/strata/kernels/glm_fast.hpp").read_text())
        start = GENERATE.index("const auto multi_step = [&] {")
        end = GENERATE.index("Gen* only = gens.size()", start)
        body = GENERATE[start:end]
        self.assertIn('env_num("STRATA_GLM_DFMULTI_DEPTH", 2.0)', GENERATE)
        self.assertIn("depth_cap = active_count <= 1 ? dfmulti_solo_depth", body)
        self.assertIn("active_count == 2 ? dfmulti_pair_depth : dfmulti_depth", body)
        self.assertNotIn("constexpr int window_depth = 2", body)

    def test_scheduler_requests_configured_window_and_reports_actual_work(self):
        start = GENERATE.index("// In DFMULTI, draft proposals remain conversation-local")
        end = GENERATE.index("Gen* only = gens.size()", start)
        body = GENERATE[start:end]
        for field in ("window_depth=", "target_rows=", "committed_prefix=", "accepted_drafts=",
                      "serial_target_calls_avoided="):
            self.assertIn(field, body, f"missing truthful window trace field {field}")

    def test_single_active_request_uses_configurable_deeper_window(self):
        self.assertIn("n_contexts < 1", ROWS,
                      "multi-window API must permit a one-context DFlash window")
        self.assertIn('env_num("STRATA_GLM_DFMULTI_SOLO_DEPTH", 8.0)', GENERATE,
                      "solo DFMULTI needs its own deeper default window")
        self.assertIn('env_num("STRATA_GLM_DFMULTI_PAIR_DEPTH", 4.0)', GENERATE,
                      "two active requests should use the four-row-per-request depth")
        start = GENERATE.index("const auto multi_step = [&] {")
        end = GENERATE.index("Gen* only = gens.size()", start)
        body = GENERATE[start:end]
        self.assertIn("depth_cap = active_count <= 1", body,
                      "only the solo depth applies when one request is active")
        self.assertIn("active_count == 2 ? dfmulti_pair_depth : dfmulti_depth", body,
                      "use depth four for two requests and the normal shared depth above two")
        self.assertIn("if (!ragged.empty())", body,
                      "any set of eligible contexts - one or many - batches into the window call")

    def test_drafter_state_is_swapped_before_prompt_captures(self):
        start = GENERATE.index("const auto gen_admit = [&]")
        end = GENERATE.index("const auto admit_pending = [&]", start)
        body = GENERATE[start:end]
        self.assertLess(body.index("dflash_ctx_swap"), body.index("prep_prompt("),
                        "the drafter's state must follow the conversation before the prompt path captures into it")

    def test_solo_and_pair_windows_draft_ahead_of_their_own_target_call(self):
        start = GENERATE.index("// In DFMULTI, draft proposals remain conversation-local")
        end = GENERATE.index("Gen* only = gens.size()", start)
        body = GENERATE[start:end]
        self.assertIn("dflash_draft_start", body,
                      "the next window's drafter pass must be queued before the current target work")
        self.assertIn("dflash_draft_finish", body,
                      "the queued drafter pass must be waited for when its drafts are needed")
        # a request whose drafts ran out this window re-proposes at the top of the next step, not synchronously
        # inside the commit path
        self.assertIn("dflash_draft_queued", body,
                      "the scheduler must track which conversation's drafter pass is in flight")
        self.assertIn("draft_in_flight", body,
                      "one drafter pass at a time: the drafter's pending-draft state is global")
        # warm matched A/B measured neutral on the one-GPU rig; opt-in, default off
        self.assertIn('env_num("STRATA_GLM_DFMULTI_AHEAD", 0.0)', body,
                      "draft-ahead must be opt-in on the one-GPU rig (measured neutral, not a default)")
        self.assertIn("draft_ahead_on && !draft_in_flight", body,
                      "the kick loop must be gated on the opt-in flag")

    def test_ragged_window_api_carries_per_context_row_counts(self):
        self.assertIn("multi_window_rows", HEADER,
                      "missing the per-context row-count window API (ragged continuous batching)")
        self.assertIn("commit_window_rows", HEADER,
                      "missing the per-context commit for ragged windows")

    def test_rows_bind_window_lays_out_rows_per_context(self):
        self.assertIn("rows_bind_window_rows", ROWS,
                      "ragged binding must place context c's rows after the contexts before it, row-major")
        self.assertIn("rows_total", ROWS,
                      "ragged binding must reject windows whose total rows exceed the arena")

    def test_scheduler_sends_ragged_windows_when_partial_readiness(self):
        start = GENERATE.index("// In DFMULTI, draft proposals remain conversation-local")
        end = GENERATE.index("Gen* only = gens.size()", start)
        body = GENERATE[start:end]
        self.assertIn("multi_window_rows", body,
                      "a partially-ready step must still batch: eligible contexts take their depth, others one row")


if __name__ == "__main__":
    unittest.main()
