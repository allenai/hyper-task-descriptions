import functools
import logging
from typing import Any, Dict, List

import datasets
import seqio
import t5
import tensorflow as tf
from catwalk.tasks import TASKS, EleutherTask, InstanceFormat
from datasets import Dataset
from t5.evaluation import metrics as mt

from hyper_task_descriptions.hf_vocab import HuggingfaceVocabulary
from hyper_task_descriptions.seqio_tasks import utils

logger = logging.getLogger(__name__)


def convert_eleuther_to_seqio(task: EleutherTask, split: str) -> Dataset:
    # Ranked classification
    if InstanceFormat.RANK_CLASSIFICATION in task.instance_conversions:

        def map_fn(instance: Dict[str, Any]):
            rank_instance = task.instance_as_rank_classification(instance)
            inputs, targets = rank_instance.choices[rank_instance.correct_choice]
            answer_choices = [ch[1] for ch in rank_instance.choices]

            # TODO: clarify what is happening with hyper_inputs.
            hyper_inputs = "<placeholder>"  # for now
            return {
                "inputs": inputs,
                "hyper_inputs": hyper_inputs,
                "targets": targets,
                "answer_choices": answer_choices,
            }

        def filter_fn(instance: Dict[str, Any]):
            return len(instance["inputs"]) > 0 and len(instance["targets"]) > 0

        dataset = task.get_split(split).inner
        original_columns = dataset.column_names
        dataset = dataset.map(map_fn).filter(filter_fn)
        # map keeps original columns, remove them
        return dataset.remove_columns(
            set(original_columns) - {"inputs", "hyper_inputs", "targets", "answer_choices"}
        )
    else:
        raise RuntimeError("Currently only works for rank classification.")


def get_eleuther_tf_dataset(
    split: str, shuffle_files: bool, seed: int, task: EleutherTask, split_mapping: Dict[str, str]
):
    # HF datasets does not support file-level shuffling
    del shuffle_files, seed
    dataset = convert_eleuther_to_seqio(task, split_mapping[split])
    return utils.hf_dataset_to_tf_dataset(dataset)


def get_dataset_splits(task: EleutherTask):
    # Note: `task.inner_task` downloads the data.
    return task.inner_task.dataset.keys()


def strip_whitespace(output_or_target, example=None, is_target=False):
    """Cached tasks from promptsource all have a leading space on the ground-truth targets."""
    return output_or_target.strip()


def get_eleuther_fixed_answer_choices(task: EleutherTask):
    # Ugh. This isn't really great, because catwalk already knows the string label.
    for feature in list(task.inner_task.dataset.values())[0].info.features.values():
        if isinstance(feature, datasets.features.features.ClassLabel):
            return feature.names
    return []


def maybe_get_class_id_postprocessor(task: EleutherTask):

    fixed_answer_choices = get_eleuther_fixed_answer_choices(task)

    if fixed_answer_choices:

        def postprocess_fn(output_or_target, example=None, is_target=False):
            output_or_target = strip_whitespace(output_or_target)
            return t5.data.postprocessors.string_label_to_class_id(
                output_or_target, label_classes=fixed_answer_choices
            )

        return postprocess_fn

    else:
        return strip_whitespace


def add_eleuther_task(task_name: str):
    logger.info(f"Adding '{task_name}' to seqio")
    task = TASKS[task_name]
    assert isinstance(task, EleutherTask)  # For now.

    # TODO: Fix.
    if InstanceFormat.RANK_CLASSIFICATION in task.instance_conversions:
        metrics = [
            mt.accuracy
        ]  # TODO: catwalk eleuther has f1, precision, recall too. Get t5 equivalent.
    else:
        metrics = [mt.accuracy]

    # TODO: do we want to restrict splits?
    dataset_splits = get_dataset_splits(task)
    split_mapping = {k: k for k in dataset_splits}

    dataset_fn = functools.partial(
        get_eleuther_tf_dataset,
        seed=None,
        task=task,
        split_mapping=split_mapping,
    )
    data_source = seqio.FunctionDataSource(
        dataset_fn,
        splits=list(split_mapping.keys()),
        num_input_examples={s: task.inner_task.dataset.num_rows[s] for s in split_mapping.keys()},
    )
    # use my unique vocab instead.
    t5_vocab = HuggingfaceVocabulary("t5-base")
    roberta_vocab = HuggingfaceVocabulary("roberta-base")

    output_features = {
        "inputs": seqio.Feature(t5_vocab, add_eos=False, dtype=tf.int32),
        "hyper_inputs": seqio.Feature(roberta_vocab, add_eos=True, dtype=tf.int32),
        "targets": seqio.Feature(t5_vocab, add_eos=True, dtype=tf.int32),
    }

    preprocessors = [
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
        seqio.CacheDatasetPlaceholder(required=False),
    ]

    # Add train and normal eval tasks
    logger.info(f'Adding {"eleuther::" + task_name} to seqio task registry')
    seqio.TaskRegistry.add(
        "eleuther::" + task_name,
        data_source,
        preprocessors=preprocessors,
        output_features=output_features,
        metric_fns=metrics,
        postprocess_fn=maybe_get_class_id_postprocessor(task),
    )

    # Add rank classification eval task
    if InstanceFormat.RANK_CLASSIFICATION in task.instance_conversions:
        rank_classification_preprocessor = functools.partial(
            t5.data.preprocessors.rank_classification,
            inputs_fn=lambda ex: tf.fill((len(ex["answer_choices"]),), ex["inputs"]),
            targets_fn=lambda ex: ex["answer_choices"],
            is_correct_fn=lambda ex: tf.equal(
                ex["answer_choices"], tf.strings.strip(ex["targets"])
            ),
            weight_fn=lambda ex: 1.0,
            passthrough_feature_keys=["hyper_inputs"],
        )
        fixed_choices = get_eleuther_fixed_answer_choices(task)
        num_classes = len(fixed_choices) if fixed_choices else None

        logger.info(f'Adding {"eleuther::" + task_name + "_score_eval"} to seqio task registry')
        seqio.TaskRegistry.add(
            "eleuther::" + task_name + "_score_eval",
            data_source,
            preprocessors=[rank_classification_preprocessor] + preprocessors,
            output_features=output_features,
            metric_fns=[
                functools.partial(
                    t5.evaluation.metrics.rank_classification, num_classes=num_classes
                )
            ],
            postprocess_fn=t5.data.postprocessors.rank_classification,
        )


def create_catwalk_mixture_list(task_names: List[str]):
    for task_name in task_names:
        add_eleuther_task(task_name)

    seqio.MixtureRegistry.add(
        "catwalk_eleuther",
        [
            task
            for task in seqio.TaskRegistry.names()
            if task.startswith("eleuther::") and not task.endswith("_score_eval")
        ],
        default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
    )

    seqio.MixtureRegistry.add(
        "catwalk_eleuther_score_eval",
        [
            task
            for task in seqio.TaskRegistry.names()
            if task.startswith("eleuther::") and task.endswith("_score_eval")
        ],
        default_rate=functools.partial(seqio.mixing_rate_num_examples, maximum=500_000),
    )


def get_eleuther_rank_classification_tasks():
    task_names = []
    for name, task in TASKS.items():
        if isinstance(task, EleutherTask):
            if InstanceFormat.RANK_CLASSIFICATION in task.instance_conversions:
                task_names.append(name)
    return task_names


create_catwalk_mixture_list(get_eleuther_rank_classification_tasks())  # ["cola", "mnli"])
