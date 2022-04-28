import functools

import seqio

# --- Seqio ---
seqio.add_global_cache_dirs(
    ["gs://hamishi-tpu-bucket/t0_data/data", "gs://hamishi-tpu-bucket/t0_data/"]
)

TaskRegistry = seqio.TaskRegistry

T5X_T0_EVAL = {
    # COPA
    "super_glue_copa_C1_or_C2_premise_so_because_",
    "super_glue_copa_best_option",
    "super_glue_copa_cause_effect",
    "super_glue_copa_choose",
    "super_glue_copa_exercise",
    "super_glue_copa_i_am_hesitating",
    "super_glue_copa_more_likely",
    "super_glue_copa_plausible_alternatives",
    "super_glue_copa__As_a_result_C1_or_C2_",
    "super_glue_copa__What_could_happen_next_C1_or_C2_",
    "super_glue_copa__which_may_be_caused_by",
    "super_glue_copa__why_C1_or_C2",
    # HellaSwag
    "hellaswag_Predict_ending_with_hint",
    "hellaswag_Randomized_prompts_template",
    "hellaswag_complete_first_then",
    "hellaswag_if_begins_how_continues",
    # Story Cloze
    "story_cloze_2016_Answer_Given_options",
    "story_cloze_2016_Choose_Story_Ending",
    "story_cloze_2016_Movie_What_Happens_Next",
    "story_cloze_2016_Novel_Correct_Ending",
    "story_cloze_2016_Story_Continuation_and_Options",
    # ANLI
    *[
        task
        for anli_round in ["r1", "r2", "r3"]
        for task in [
            f"anli_GPT_3_style_{anli_round}",
            f"anli_MNLI_crowdsource_{anli_round}",
            f"anli_always_sometimes_never_{anli_round}",
            f"anli_based_on_the_previous_passage_{anli_round}",
            f"anli_can_we_infer_{anli_round}",
            f"anli_claim_true_false_inconclusive_{anli_round}",
            f"anli_consider_always_sometimes_never_{anli_round}",
            f"anli_does_it_follow_that_{anli_round}",
            f"anli_does_this_imply_{anli_round}",
            f"anli_guaranteed_true_{anli_round}",
            f"anli_guaranteed_possible_impossible_{anli_round}",
            f"anli_justified_in_saying_{anli_round}",
            f"anli_must_be_true_{anli_round}",
            f"anli_should_assume_{anli_round}",
            f"anli_take_the_following_as_truth_{anli_round}",
        ]
    ],
    # CB
    "super_glue_cb_GPT_3_style",
    "super_glue_cb_MNLI_crowdsource",
    "super_glue_cb_always_sometimes_never",
    "super_glue_cb_based_on_the_previous_passage",
    "super_glue_cb_can_we_infer",
    "super_glue_cb_claim_true_false_inconclusive",
    "super_glue_cb_consider_always_sometimes_never",
    "super_glue_cb_does_it_follow_that",
    "super_glue_cb_does_this_imply",
    "super_glue_cb_guaranteed_true",
    "super_glue_cb_guaranteed_possible_impossible",
    "super_glue_cb_justified_in_saying",
    "super_glue_cb_must_be_true",
    "super_glue_cb_should_assume",
    "super_glue_cb_take_the_following_as_truth",
    # RTE
    "super_glue_rte_GPT_3_style",
    "super_glue_rte_MNLI_crowdsource",
    "super_glue_rte_based_on_the_previous_passage",
    "super_glue_rte_can_we_infer",
    "super_glue_rte_does_it_follow_that",
    "super_glue_rte_does_this_imply",
    "super_glue_rte_guaranteed_true",
    "super_glue_rte_justified_in_saying",
    "super_glue_rte_must_be_true",
    "super_glue_rte_should_assume",
    # WSC
    "super_glue_wsc.fixed_GPT_3_Style",
    "super_glue_wsc.fixed_I_think_they_mean",
    "super_glue_wsc.fixed_Who_or_what_is_are",
    "super_glue_wsc.fixed_by_p_they_mean",
    "super_glue_wsc.fixed_does_p_stand_for",
    "super_glue_wsc.fixed_does_the_pronoun_refer_to",
    "super_glue_wsc.fixed_in_other_words",
    "super_glue_wsc.fixed_p_is_are_r",
    "super_glue_wsc.fixed_replaced_with",
    "super_glue_wsc.fixed_the_pronoun_refers_to",
    # Winogrande
    "winogrande_winogrande_xl_Replace",
    "winogrande_winogrande_xl_does_underscore_refer_to",
    "winogrande_winogrande_xl_fill_in_the_blank",
    "winogrande_winogrande_xl_stand_for",
    "winogrande_winogrande_xl_underscore_refer_to",
    # WiC
    "super_glue_wic_GPT_3_prompt",
    "super_glue_wic_GPT_3_prompt_with_label",
    "super_glue_wic_affirmation_true_or_false",
    "super_glue_wic_grammar_homework",
    "super_glue_wic_polysemous",
    "super_glue_wic_question_context",
    "super_glue_wic_question_context_meaning",
    "super_glue_wic_question_context_meaning_with_label",
    "super_glue_wic_same_sense",
    "super_glue_wic_similar_sense",
}
seqio.MixtureRegistry.add(
    "t5x_t0_eval",
    [
        task
        for task in seqio.TaskRegistry.names()
        if task.endswith("_score_eval") and task.split("_score_eval")[0] in T5X_T0_EVAL
    ],
    default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
)  # eval mixture does not need to be capped

# --- Improve sharding ---

# def fully_sharded_logical_axis_rules() -> LogicalAxisRules:
#     """Fully sharded rules for P5X model in terms of logical axes names."""
#     return (
#       ('batch', 'data'),
#       ('vocab', 'model'),
#       ('mlp', 'model'),
#       ('heads', 'model'),
#       ('joined_kv', 'model'),
#       ('kv', None),
#       ('embed', 'model'),
#       ('embed', 'data'),
#       ('relpos_buckets', None),
#       ('length', None),
#       ('layers', None),
#       ('stack', None),
#     )
