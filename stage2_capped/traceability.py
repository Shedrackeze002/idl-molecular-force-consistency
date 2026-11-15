"""Traceability matrix for the Stage 2 study: every verification bullet of the study
specification's section 7 (verification gates before any real-model launch) and the named
tests that cover it. `tests/test_traceability.py` checks that every bullet is present, that
every named test exists, and that `TRACEABILITY.md` is exactly what `render()` produces.

A bullet that no test can cover (a statement a launch record must make) carries a `record`
note instead and is marked as such, so the gap is visible rather than implied.
"""
import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent / "tests"

# The section 7 bullets, in the specification's order.
BULLET_IDS = (
    "E1-diff-gates", "E2-bitwise-at-huge-caps", "E3-prefix-gate", "E4-stage0-comparator",
    "E5-stage0-hard-stop", "E6-mismatch-procedure", "E7-stage-gating",
    "C1-cap-bound", "C2-backtracking", "C3-binding-recorded", "C4-exceptions-propagate",
    "C5-determinism", "C6-refusals",
    "R1-rules", "R2-admission",
)

RC, RF, RA = "test_relax_capped.py", "test_restore_fork.py", "test_admission.py"
PG, PC, S0, SG, DR, LI, RU = ("test_prefix_gate.py", "test_pass_copy.py", "test_stage0_compare.py",
                              "test_stage_gate.py", "test_driver.py", "test_launch_io.py", "test_rules.py")
DY, FZ, TR = "test_dry_run.py", "test_frozen_files.py", "test_traceability.py"
DV, MU, RS = "test_driver_review.py", "test_review_mutants.py", "test_row_schema.py"

MATRIX = [
    {"id": "E1-diff-gates",
     "bullet": "Diff gate: the capped relaxer equals relax.py plus an allow-list of hunks (frozen result fields only); "
               "the same gate for the restore fork and the copied runner pass.",
     "tests": [(RC, "test_diff_gate_passes_on_the_committed_files"), (RC, "test_diff_gate_catches_an_unlisted_change"),
               (RC, "test_diff_gate_catches_a_changed_frozen_file"),
               (RC, "test_diff_gate_catches_a_missing_and_an_extra_hunk"),
               (RC, "test_every_listed_hunk_has_a_purpose_and_allowed_categories"),
               (RF, "test_diff_gate_passes_on_the_committed_files"),
               (RF, "test_the_allowlist_pins_the_frozen_file_and_lists_only_three_categories"),
               (RF, "test_diff_gate_catches_an_unlisted_change"), (RF, "test_diff_gate_catches_each_kind_of_edit"),
               (RF, "test_diff_gate_catches_a_changed_frozen_file"),
               (RF, "test_diff_gate_catches_a_missing_hunk_and_a_bad_category"),
               (PC, "test_diff_gate_passes_on_the_committed_files"), (PC, "test_the_allowlist_pins_the_runner_and_the_excerpt"),
               (PC, "test_diff_gate_catches_each_kind_of_edit"),
               (PC, "test_diff_gate_catches_a_change_inside_the_frozen_excerpt_and_outside_it"),
               (PC, "test_the_line_range_must_lie_inside_the_frozen_file"),
               (PC, "test_a_target_without_a_line_range_behaves_as_before")]},
    {"id": "E2-bitwise-at-huge-caps",
     "bullet": "c in {None, infinity, 1e300, 1e12, 1e6} gives bitwise-identical results on synthetic fixtures, including "
               "budget exhaustion, loose tolerance and every refusal mode; recorder on equals recorder off.",
     "tests": [(RC, "test_bitwise_equal_to_frozen_at_huge_caps"), (RC, "test_scenarios_cover_the_outcome_modes"),
               (RC, "test_default_cap_is_the_control"), (RC, "test_recorder_on_equals_recorder_off"),
               (RC, "test_recorder_mutation_cannot_change_the_run"), (RC, "test_recorder_errors_propagate"),
               (PC, "test_without_a_cap_the_copied_pass_reproduces_the_frozen_rows"),
               (DR, "test_the_original_run_equals_the_frozen_evaluator_at_no_cap")]},
    {"id": "E3-prefix-gate",
     "bullet": "Prefix-equality gate: the control trace predicts each capped run's first binding trial; the capped run "
               "equals its control up to it, and entirely if nothing binds; a c = 1.0 duplicate of the control is reported.",
     "tests": [(PG, "test_capped_run_follows_the_control_prediction"), (PG, "test_unbound_capped_run_equals_the_control_entirely"),
               (PG, "test_c1p00_duplicate_of_the_control_is_detected"),
               (PG, "test_late_binding_fixture_binds_at_the_predicted_attempts"),
               (PG, "test_prediction_uses_all_trials_including_rejected_ones"),
               (PG, "test_gate_flags_a_capped_run_that_diverges_early"), (PG, "test_gate_flags_a_wrong_cap_claim"),
               (PG, "test_gate_flags_a_cap_that_never_bound_but_reports_capped"),
               (PG, "test_gate_flags_a_capped_run_that_stops_before_the_binding_attempt"),
               (PG, "test_gate_flags_a_missing_candidate_change_at_binding"),
               (DR, "test_the_prefix_gate_stops_a_run_whose_control_trace_disagrees"),
               (DR, "test_a_capped_stage_runs_against_the_control_index_and_passes_the_prefix_gate"),
               (DR, "test_a_missing_control_is_a_hard_stop"), (DR, "test_a_control_stage_completes_and_summarizes_binding"),
               (DR, "test_a_control_run_from_far_away_reports_that_c1p00_would_bind")]},
    {"id": "E4-stage0-comparator",
     "bullet": "Stage 0 comparator and loader for the 50 saved rows and the 30 saved relaxation files: a self-compare "
               "passes and a perturbed field fails.",
     "tests": [(S0, "test_the_real_saved_results_load_as_50_and_30"), (S0, "test_self_compare_of_the_real_results_passes"),
               (S0, "test_any_perturbed_original_field_of_the_real_results_fails"),
               (S0, "test_any_perturbed_reference_field_of_the_real_results_fails"),
               (S0, "test_first_difference_is_bit_exact_and_type_strict"), (S0, "test_each_hard_stop_field_is_compared"),
               (S0, "test_original_final_positions_are_compared_bitwise"),
               (S0, "test_reference_comparison_covers_every_saved_field_and_ignores_extra_new_fields"),
               (S0, "test_the_loader_accepts_a_complete_fabricated_set_and_refuses_damage"),
               (S0, "test_the_loader_refuses_duplicate_starts_and_unequal_start_sets"),
               (S0, "test_the_reference_loader_refuses_a_missing_or_extra_relaxation"),
               (S0, "test_incomplete_new_results_are_refused"), (S0, "test_the_saved_file_hashes_cover_all_80_results"),
               (S0, "test_reference_comparison_checks_the_saved_category_and_stop_reason"),
               (S0, "test_a_changed_reference_category_alone_fails_on_the_real_results"),
               (S0, "test_the_real_saved_reference_results_carry_a_category_and_stop_reason_for_all_30")]},
    {"id": "E5-stage0-hard-stop",
     "bullet": "Stage 0 hard stop: the 80 control runs match the saved results exactly (category, stop reason, step count, "
               "force-evaluation count, flags, bitwise final positions; for the 30 reference runs also steps and force maximums).",
     "tests": [(DR, "test_stage_zero_passes_when_every_run_reproduces_the_saved_result"),
               (DR, "test_stage_zero_stops_at_the_first_mismatch_and_writes_the_evidence"),
               (DR, "test_an_incomplete_stage_zero_is_not_a_pass"), (SG, "test_a_clean_stage_zero_passes_only_after_all_80_runs"),
               (DR, "test_stage_zero_is_the_accepted_order_and_matches_the_comparator_schedule"),
               (DR, "test_the_fixed_schedules_have_the_spec_run_counts"),
               (DR, "test_stage_zero_stops_when_only_a_saved_reference_category_or_stop_reason_differs"),
               (DV, "test_stage_zero_refuses_a_controller_that_expects_fewer_than_all_80_runs"),
               (DV, "test_the_stage_zero_manifest_holds_the_verdict_the_hashes_and_one_match_line_per_task"),
               (DY, "test_a_complete_launch_runs_every_stage_through_supervised_workers")]},
    {"id": "E6-mismatch-procedure",
     "bullet": "Mismatch procedure: fail fast at the first mismatch; one pre-authorized re-run of that single task with the "
               "frozen relax; frozen differs = runtime drift, stop; frozen matches but c = infinity does not = code error, stop; "
               "any tolerance or further re-run needs a new reviewed record.",
     "tests": [(SG, "test_the_first_mismatch_halts_and_asks_for_the_single_frozen_rerun"),
               (SG, "test_the_rerun_outcome_decides_drift_or_code_error_and_both_stop"),
               (SG, "test_only_one_rerun_and_only_of_the_first_mismatching_task"),
               (SG, "test_a_result_cannot_be_recorded_without_authorization"),
               (SG, "test_an_authorized_rerun_that_crashes_stays_spent"),
               (DR, "test_a_refused_frozen_rerun_makes_zero_force_calls"),
               (SG, "test_no_tolerance_a_single_bit_problem_is_a_mismatch"),
               (SG, "test_stage_zero_verdicts_are_the_rules_verdicts"),
               (DR, "test_the_single_frozen_rerun_decides_runtime_drift_when_frozen_also_differs"),
               (DR, "test_a_code_error_is_told_apart_from_runtime_drift"),
               (RU, "test_r_control_passes_only_on_a_complete_exact_match"),
               (RU, "test_r_control_first_mismatch_authorizes_one_frozen_rerun"),
               (RU, "test_r_control_runtime_drift_and_code_error_both_stop"),
               (RU, "test_r_control_refuses_any_continuation_beyond_the_pre_authorized_rerun"),
               (DY, "test_a_stage_zero_mismatch_stops_and_the_single_frozen_rerun_finds_a_code_error")]},
    {"id": "E7-stage-gating",
     "bullet": "Stage gating: Stage 0b needs Stage 0; Stage 1 needs Stage 0; Stage 2 needs Stage 1, Stage 0b and the A0 replay; "
               "a stage runs once with no retry or resume; the gating and mismatch state machine is tested. "
               "(The A0 replay itself runs in launch A0.)",
     "tests": [(SG, "test_the_happy_path_runs_in_the_spec_order"), (SG, "test_a_stage_cannot_start_before_its_prerequisites"),
               (SG, "test_stage_two_needs_stage_one_stage_0b_and_the_a0_replay"),
               (SG, "test_a_failed_a0_blocks_everything_and_a_failed_stage_zero_blocks_the_rest"),
               (SG, "test_stage_one_does_not_need_stage_0b"), (SG, "test_no_stage_runs_twice_and_none_resumes"),
               (SG, "test_only_one_stage_runs_at_a_time_and_finish_needs_a_running_stage"),
               (SG, "test_exhaustively_no_order_violation_is_possible"),
               (SG, "test_stage_two_still_needs_the_replay_flag_if_the_ledger_is_set_up_by_hand"),
               (DR, "test_the_ledger_gates_a_stage_before_anything_is_written"),
               (DR, "test_the_ledger_records_completion_and_failure"),
               (DR, "test_a_killed_launch_is_never_resumed_a_new_launch_gets_a_new_root"),
               (LI, "test_a_launch_root_is_new_and_never_reused"),
               (DY, "test_a_completed_stage_cannot_be_run_again_in_the_same_root"),
               (DY, "test_an_overrunning_stage_is_terminated_keeps_its_rows_and_blocks_the_rest"),
               (DY, "test_a_worker_that_ignores_sigterm_is_killed_and_the_supervisor_marks_the_stage"),
               (DY, "test_kill_then_new_launch_leaves_the_old_root_untouched"),
               (DY, "test_the_storage_guard_stops_a_worker_and_the_stage_is_marked")]},
    {"id": "C1-cap-bound",
     "bullet": "max over atoms of the displacement is at most c(1 + 1e-12) on every trial, accepted and rejected, on fixtures with s < 1.",
     "tests": [(RC, "test_every_trial_obeys_the_cap"), (RC, "test_rejected_trials_obey_the_cap_too"),
               (RC, "test_cap_binds_exactly_when_s_f_exceeds_c"), (RC, "test_effective_scale_formula_and_units"),
               (RC, "test_no_step_test_runs_on_the_uncapped_step"),
               (DR, "test_a_capped_run_records_its_cap_and_never_moves_an_atom_further_than_it")]},
    {"id": "C2-backtracking",
     "bullet": "Backtracking: on a stiff harmonic fixture no two consecutive trials from one parent have identical candidates.",
     "tests": [(RC, "test_backtracking_shrinks_the_real_step_not_s"), (RC, "test_growth_doubles_s_not_s_eff"),
               (RC, "test_accepted_step_size_is_the_applied_scale"), (RC, "test_floor_edge_is_documented_behavior")]},
    {"id": "C3-binding-recorded",
     "bullet": "Per-run cap binding is recorded and tested.",
     "tests": [(RC, "test_cap_record_summary_matches_the_trace"), (RC, "test_control_summary_has_no_capped_trials_and_none_cap"),
               (RC, "test_a_bound_cap_changes_the_far_start_run"),
               (PC, "test_a_binding_cap_changes_the_run_and_the_row_records_it"),
               (DR, "test_the_control_fields_are_assigned_to_the_right_columns")]},
    {"id": "C4-exceptions-propagate",
     "bullet": "A refusal battery shows that an unrelated exception propagates.",
     "tests": [(RC, "test_unrelated_exceptions_propagate_under_a_cap"), (RC, "test_declared_refusals_are_recorded_under_a_cap")]},
    {"id": "C5-determinism",
     "bullet": "A subset run twice is bitwise identical.",
     "tests": [(RC, "test_determinism_rerun_is_bitwise_identical")]},
    {"id": "C6-refusals",
     "bullet": "A cap outside the allowed set and a batch size other than 1 are refused.",
     "tests": [(RC, "test_bad_caps_are_refused"), (RC, "test_cap_validation_values"),
               (RC, "test_bad_cap_is_refused_before_any_force_call"), (RC, "test_batch_size_other_than_one_is_refused"),
               (RC, "test_batch_of_one_and_unbatched_agree"),
               (DR, "test_prevalidation_refuses_a_batched_start_before_any_force_call"),
               (DR, "test_a_batch_of_one_and_a_degenerate_start_are_refused_before_any_force_call")]},
    {"id": "R1-rules",
     "bullet": "R-control, R-adequacy, R-floor, R-direction, the placebo check, R-sweep and R-attribution are coded in rules.py "
               "and unit-tested on synthetic outcome tables, including a seeded property test.",
     "tests": [(RU, "test_clean_table_states_a_direction_for_consistency"), (RU, "test_gate_failure_is_uninformative_not_a_null"),
               (RU, "test_floor_items_are_left_out_and_listed"), (RU, "test_support_failure_is_uninformative"),
               (RU, "test_direction_is_two_sided_and_the_other_arm_vetoes_it_either_way"),
               (RU, "test_placebo_flags_a_cap_where_seed_contrasts_look_like_the_main_contrast"),
               (RU, "test_a_placebo_flagged_cap_never_contributes_to_a_stated_direction"),
               (RU, "test_one_evaluable_cap_is_a_single_cap_observation_whether_or_not_direction_holds"),
               (RU, "test_opposite_signs_across_caps_give_no_direction"),
               (RU, "test_direction_stated_needs_the_common_set_to_hold_too"),
               (RU, "test_attribution_supported_with_two_qualifying_runs"),
               (RU, "test_attribution_counts_lambda_one_runs_but_labels_them_exploratory"),
               (RU, "test_property_every_table_gets_exactly_one_label_and_the_rules_hold"),
               (RU, "test_property_counterfactual_removing_a_placebo_flagged_cap_changes_nothing"),
               (RU, "test_no_record_uses_effect_or_significant")],
     "record": "The scientific reviewer's sign-off of rules.py, with the file's SHA-256 (5c52660c78427166...), goes into "
               "launch record B before Stage 0; a statement, not a test."},
    {"id": "R2-admission",
     "bullet": "The seven checkpoint-identity hash checks and the pin policy are tested without loading any model.",
     "tests": [(RA, "test_identical_pins_pass_with_nothing_recorded"),
               (RA, "test_python_platform_and_template_drift_is_recorded_not_fatal"),
               (RA, "test_every_cpu_fact_except_the_platform_string_must_be_equal"), (RA, "test_any_required_difference_is_refused"),
               (RA, "test_every_identity_check_can_fail"),
               (RA, "test_an_accepted_model_must_match_its_independent_anchor_not_just_its_own_records"),
               (RA, "test_all_seven_identities_admit_from_the_saved_files"),
               (RA, "test_the_five_anchors_equal_the_records_and_appear_in_independent_files"),
               (RA, "test_admission_loads_no_model_and_never_names_the_test_partition"),
               (RA, "test_the_dataset_pin_must_match_and_must_not_name_the_test_partition"),
               (RF, "test_only_a_real_pin_policy_that_ran_and_found_no_problem_lets_the_restore_continue"),
               (RF, "test_the_fork_adds_no_model_loading_and_no_test_partition_access")]},
]


# The plan's M6 acceptance items that are not spec section 7 bullets.
PLAN_ITEMS = [
    {"id": "M6-traceability", "item": "Every spec section 7 bullet maps to a named test.",
     "tests": [(TR, "test_every_section_7_bullet_is_in_the_matrix_once_and_in_order"), (TR, "test_every_named_test_exists"),
               (TR, "test_the_markdown_file_is_what_render_produces")]},
    {"id": "M6-frozen-files", "item": "No frozen file changed (hash check).",
     "tests": [(FZ, "test_each_frozen_file_has_its_recorded_hash"),
               (FZ, "test_the_source_pin_matches_the_accepted_launch_pin_in_every_required_field")]},
    {"id": "M6-dry-run",
     "item": "Full synthetic driver dry run: timeout trip with partial rows kept, storage-guard trip, Stage 0 gating, kill-then-new-launch.",
     "tests": [(DY, "test_a_complete_launch_runs_every_stage_through_supervised_workers"),
               (DY, "test_an_overrunning_stage_is_terminated_keeps_its_rows_and_blocks_the_rest"),
               (DY, "test_a_worker_that_ignores_sigterm_is_killed_and_the_supervisor_marks_the_stage"),
               (DY, "test_the_storage_guard_stops_a_worker_and_the_stage_is_marked"),
               (DY, "test_a_stage_zero_mismatch_stops_and_the_single_frozen_rerun_finds_a_code_error"),
               (DY, "test_kill_then_new_launch_leaves_the_old_root_untouched")]},
    {"id": "M4-review-fixes",
     "item": "Findings of the independent M4 engineering review: reference category compared (F01), the one re-run "
             "authorized first (F02), ledger and schedule enforced inside run_stage (F03), stage manifest (F04), launch "
             "root marker (F05), launch-record and procedure binding (F06), control-index refusals (F07), every arm "
             "built before any force call (F08), and the tests for the mutants that survived (F09).",
     "tests": [(S0, "test_a_changed_reference_category_alone_fails_on_the_real_results"),
               (DR, "test_a_refused_frozen_rerun_makes_zero_force_calls"),
               (DV, "test_a_stage_needs_a_ledger_and_the_fixed_schedule_unless_a_test_flag_is_set"),
               (DV, "test_a_finished_stage_writes_a_manifest_and_the_completion_record_names_its_hash"),
               (DV, "test_a_context_needs_a_root_made_for_its_own_launch_id"),
               (DV, "test_the_launch_records_must_have_the_hash_the_launch_record_names"),
               (DV, "test_a_complete_control_root_loads_and_is_refused_unless_custom_runs_are_allowed"),
               (DV, "test_a_failing_second_arm_stops_the_stage_before_any_force_call_and_before_the_ledger_starts"),
               (DV, "test_the_incomplete_marker_lists_only_runs_that_passed_every_gate"),
               (DV, "test_the_storage_reserve_comes_from_the_trace_record_size_at_the_budget"),
               (LI, "test_a_crash_before_the_final_name_leaves_only_a_partial_file"),
               (LI, "test_the_supervisor_marks_a_killed_stage_once_and_lists_only_finished_rows"),
               (RS, "test_converged_follows_the_outcome_category_in_both_directions"),
               (MU, "test_the_prefix_gate_runs_in_both_capped_stages"),
               (MU, "test_the_model_identity_table_agrees_with_every_model_id"),
               (MU, "test_accepted_and_attempted_travel_are_the_sums_over_the_right_trials"),
               (MU, "test_a_cap_binds_exactly_when_the_step_would_exceed_it_never_at_equality"),
               (MU, "test_a_run_is_refused_when_the_files_plus_the_reserve_would_pass_the_limit"),
               (MU, "test_the_diff_gate_reports_a_trailing_newline_difference_by_name"),
               (DV, "test_a_launch_stage_zero_needs_the_hashes_of_all_110_saved_files"),
               (DV, "test_stage_zero_refuses_a_saved_set_that_misses_a_scheduled_task_before_any_force_call"),
               (DV, "test_a_checkpoint_identity_that_is_not_plain_json_is_refused_before_any_force_call"),
               (DV, "test_a_stage_the_ledger_refuses_builds_no_arm_at_all"),
               (DV, "test_the_manifest_counts_refused_trials"),
               (DV, "test_the_production_path_runs_the_fixed_stages_with_real_inputs_and_no_test_flag"),
               (MU, "test_the_1p0_threshold_is_exactly_1p0_in_the_summary_and_in_the_row"),
               (MU, "test_a_reference_run_that_runs_out_of_budget_is_scored_with_its_own_flags"),
               (MU, "test_a_control_trace_that_is_shorter_than_its_row_is_refused"),
               (LI, "test_an_existing_folder_without_a_marker_is_never_taken_as_a_launch_root"),
               (DV, "test_a_spent_stage_folder_is_refused_before_the_ledger_starts_the_stage"),
               (MU, "test_the_0p25_and_1p0_step_fractions_are_pinned_from_both_sides"),
               (MU, "test_a_step_just_above_1p0_binds_the_c1p00_cap_in_the_row_the_summary_and_the_duplicate_flag")]},
]


def defined_tests(filename):
    """Names of the module-level test functions in a test file."""
    tree = ast.parse((TESTS / filename).read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def missing_tests():
    """(file, name) pairs the matrix names that do not exist."""
    cache, missing = {}, []
    for entry in MATRIX + PLAN_ITEMS:
        for filename, name in entry["tests"]:
            if filename not in cache:
                cache[filename] = defined_tests(filename)
            if name not in cache[filename]:
                missing.append((filename, name))
    return missing


def render():
    """The matrix as markdown (what TRACEABILITY.md must contain)."""
    lines = ["# Stage 2 traceability matrix", "",
             "Every verification bullet of the study specification's section 7 and the named tests that cover it. "
             "Generated by `stage2_capped/traceability.py` and checked by `tests/test_traceability.py`; do not edit by hand.",
             "", "| ID | Bullet | Covered by (test file :: test) |", "|---|---|---|"]
    for entry in MATRIX:
        tests = "<br>".join(f"{f}::{n}" for f, n in entry["tests"])
        note = f"<br>**Record:** {entry['record']}" if entry.get("record") else ""
        lines.append(f"| {entry['id']} | {entry['bullet']} | {tests}{note} |")
    total = sum(len(e["tests"]) for e in MATRIX)
    lines += ["", f"{len(MATRIX)} bullets, {total} test references "
                  f"({len({t for e in MATRIX for t in e['tests']})} distinct tests). "
                  "A bullet with a **Record** note also needs a statement in a launch record.",
              "", "## Plan M6 acceptance items", "", "| ID | Item | Covered by |", "|---|---|---|"]
    for item in PLAN_ITEMS:
        lines.append(f"| {item['id']} | {item['item']} | " + "<br>".join(f"{f}::{n}" for f, n in item["tests"]) + " |")
    return "\n".join(lines) + "\n"
