"""
From t-zero repo
"""
import re

import datasets
import pkg_resources
import tensorflow as tf
from promptsource.utils import removeHyphen
from tensorflow.python.data import Dataset
from tensorflow.python.data.ops.dataset_ops import (
    DatasetV2,
    MapDataset,
    RandomDataset,
    _DirectedInterleaveDataset,
)
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import array_ops, gen_stateless_random_ops, math_ops

from hyper_task_descriptions.seqio_tasks.t0_datasets_mapping import T0_DS_MAPPING


def replace_keys(prompt):
    return re.sub(r"\[[A-Za-z0-9_]+\]", lambda x: "[MASK]", prompt)


def load_prewritten_prompts():
    text = open(pkg_resources.resource_filename(__name__, "all_edited_prompts.txt"), "r").read()
    text = text.split("****************************")
    text = [t.strip().split("	||||	") for t in text]
    text = {t[0] + "_" + t[1]: replace_keys(t[2]) for t in text if len(t) > 2}
    text = {k.lower().strip(): v for k, v in text.items()}
    return text


def feature_to_spec(feature, length=False):
    if isinstance(feature, datasets.ClassLabel):
        return tf.TensorSpec(
            shape=() if not length else (None if length == -1 else length,), dtype=tf.int64
        )
    elif isinstance(feature, datasets.Value):
        return tf.TensorSpec(
            shape=() if not length else (None if length == -1 else length,),
            dtype=getattr(tf.dtypes, feature.dtype),
        )
    elif hasattr(feature, "dtype") and hasattr(feature, "shape"):
        return tf.TensorSpec(shape=feature.shape, dtype=feature.dtype)
    elif isinstance(feature, datasets.Sequence):
        return feature_to_spec(feature.feature, length=feature.length)
    elif isinstance(feature, list):
        return [feature_to_spec(f, length=length) for f in feature]
    elif isinstance(feature, dict):
        return {k: feature_to_spec(v, length=length) for k, v in feature.items()}
    else:
        raise ValueError(f"Unparseable feature type {type(feature)}")


def hf_dataset_to_tf_dataset(dataset):
    return tf.data.Dataset.from_generator(
        dataset.__iter__,
        output_signature={k: feature_to_spec(v) for k, v in dataset.features.items()},
    )


def apply_template(dataset, template):
    def map_fn(ex):
        ex = removeHyphen(ex)
        inputs_and_targets = template.apply(ex)
        answer_choices = template.get_answer_choices_list(ex)
        if len(inputs_and_targets) == 2:
            inputs, targets = inputs_and_targets
            if targets == "":
                ex = {"inputs": inputs, "targets": "<NO LABEL>"}
            else:
                ex = {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            ex = {"inputs": "", "targets": ""}

        if answer_choices:
            ex["answer_choices"] = answer_choices

        return ex

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(set(original_columns) - {"inputs", "targets", "answer_choices"})


def apply_template_split(dataset, template, dataset_name, subset_name=None):
    def map_fn(ex):
        ex = removeHyphen(ex)
        inputs_and_targets = template.apply(ex)
        answer_choices = template.get_answer_choices_list(ex)
        if len(inputs_and_targets) == 2:
            inputs, targets = inputs_and_targets
            if targets == "":
                ex = {"inputs": inputs, "targets": "<NO LABEL>"}
            else:
                ex = {"inputs": inputs, "targets": targets}
        # When template results in an empty example, template.apply returns [""]
        # Also, if the template gets split wrong, len can be > 2
        # We will filter these out later
        else:
            ex = {"inputs": "", "targets": ""}

        if answer_choices:
            ex["answer_choices"] = answer_choices

        # load prewritten prompts and grab respective one
        prompt_dict = load_prewritten_prompts()
        task_name = get_task_name(dataset_name, subset_name, template.name)
        # various fixing stuff to make task name match my prompt edits.
        if "score_eval" in task_name:
            task_name = task_name.replace("_score_eval", "")
        if "anli" in task_name:  # anli is handled weird
            task_name += "_r1"
        # new code - grab the input items and *remove them from the input text*. This is our template.
        ex["template"] = prompt_dict[task_name.lower()]
        ex["hyper_inputs"] = ""
        # counter = 0
        # TODO: check how many inputs this actually covers.
        # a simple replacement setup for now.
        # for v in og_ex.values():
        #     if isinstance(v, str) and v in ex["inputs"]:
        #         ex["template"] = ex["template"].replace(str(v), f"[{counter + 1}]")
        #         ex["hyper_inputs"] = ex["hyper_inputs"] + f"[{counter + 1}]: {str(v)}\n"
        #         counter += 1
        #     elif isinstance(v, str) and v.lower() in ex["inputs"]:
        #         ex["template"] = ex["template"].replace(str(v).lower(), f"[{counter + 1}]")
        #         ex["hyper_inputs"] = ex["hyper_inputs"] + f"[{counter + 1}]: {str(v).lower()}\n"
        #         counter += 1
        # flip due to earlier decisions made.
        ex["hyper_inputs"], ex["template"] = ex["template"], ex["hyper_inputs"]
        if subset_name is not None:
            ex["task_names"] = tf.constant(
                [T0_DS_MAPPING[dataset_name + "_" + subset_name]], dtype=tf.int32
            )
        else:
            ex["task_names"] = tf.constant([T0_DS_MAPPING[dataset_name]], dtype=tf.int32)
        return ex

    def filter_fn(ex):
        return len(ex["inputs"]) > 0 and len(ex["targets"]) > 0

    original_columns = dataset.column_names
    dataset = dataset.map(map_fn).filter(filter_fn)
    # map keeps original columns, remove them
    return dataset.remove_columns(
        set(original_columns)
        - {"inputs", "hyper_inputs", "targets", "answer_choices", "task_names"}
    )


def get_dataset_splits(dataset_name, subset_name=None):
    info = datasets.get_dataset_infos(
        dataset_name, download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS
    )
    subset_name = subset_name or list(info.keys())[0]
    return info[subset_name].splits


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(
        dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name
    )


def double_sample_from_datasets(datasets, weights=None, seed=None, stop_on_empty_dataset=False):
    """
    An altered form of tf.data.Datasets.sample_from_datasets that samples the same,
    but takes 2 samples per dataset at a time. This is to allow more more control
    over contrastive sampling.
    Seed is mandatory rn to recreate the random ds.
    """
    if seed is None:
        seed = 42

    def _skip_datasets_with_zero_weight(datasets, weights):
        datasets_and_weights = [
            (dataset, weight) for (dataset, weight) in zip(datasets, weights) if weight > 0
        ]
        return (
            zip(*datasets_and_weights) if datasets_and_weights else ([datasets[0].take(0)], [1.0])
        )

    if not datasets:
        raise ValueError("Invalid `datasets`. `datasets` should not be empty.")

    if not isinstance(weights, DatasetV2):
        if weights is None:
            # Select inputs with uniform probability.
            logits = [[1.0] * len(datasets)]

        else:
            if isinstance(weights, ops.Tensor):
                if not weights.shape.is_compatible_with([len(datasets)]):
                    raise ValueError(
                        f"Invalid `weights`. The shape of `weights` "
                        f"should be compatible with `[len(datasets)]` "
                        f"but is {weights.shape}."
                    )
            else:
                if len(datasets) != len(weights):
                    raise ValueError(
                        f"Invalid `weights`. `weights` should have the "
                        f"same length as `datasets` but got "
                        f"`len(weights)={len(weights)}` vs. "
                        f"`len(datasets)={len(datasets)}`."
                    )

            # Use the given `weights` as the probability of choosing the respective
            # input.
            if not isinstance(weights, ops.Tensor):
                datasets, weights = _skip_datasets_with_zero_weight(datasets, weights)
            weights = ops.convert_to_tensor(weights, name="weights")
            if weights.dtype not in (dtypes.float32, dtypes.float64):
                raise TypeError(
                    f"Invalid `weights`. `weights` type must be either "
                    f"`tf.float32` or `tf.float64` but is "
                    f"{weights.dtype}."
                )

            # The `stateless_multinomial()` op expects log-probabilities, as opposed
            # to weights.
            logits = array_ops.expand_dims(math_ops.log(weights, name="logits"), 0)

        # NOTE(mrry): We only specialize when `weights` is not a `Dataset`. When
        # it is a `Dataset`, it is possible that evaluating it has a side effect
        # the user depends on.
        if len(datasets) == 1:
            return datasets[0]

        def select_dataset_constant_logits(seed):
            return array_ops.squeeze(
                gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed), axis=[0, 1]
            )

        selector_input = MapDataset(
            RandomDataset(seed).batch(2),
            select_dataset_constant_logits,
            use_inter_op_parallelism=False,
        )
        selector_input_copy = MapDataset(
            RandomDataset(seed).batch(2),
            select_dataset_constant_logits,
            use_inter_op_parallelism=False,
        )

    else:
        # Use each element of the given `weights` dataset as the probability of
        # choosing the respective input.
        #
        # The `stateless_multinomial()` op expects log-probabilities, as opposed
        # to weights.
        logits_ds = weights.map(lambda *p: math_ops.log(p, name="logits"))

        def select_dataset_varying_logits(logits, seed):
            return array_ops.squeeze(
                gen_stateless_random_ops.stateless_multinomial(logits, 1, seed=seed), axis=[0, 1]
            )

        logits_and_seeds = Dataset.zip((logits_ds, RandomDataset(seed).batch(2)))
        selector_input = MapDataset(
            logits_and_seeds, select_dataset_varying_logits, use_inter_op_parallelism=False
        )
        selector_input_copy = MapDataset(
            logits_and_seeds, select_dataset_varying_logits, use_inter_op_parallelism=False
        )

    # we combine selector inputs, so when sampling we should get two of the same at a time.
    selector_input = selector_input.batch(1)
    selector_input_copy = selector_input_copy.batch(1)
    selector_input = (
        tf.data.Dataset.zip((selector_input, selector_input_copy))
        .map(lambda x, y: tf.concat((x, y), axis=0))
        .unbatch()
    )

    return _DirectedInterleaveDataset(selector_input, datasets, stop_on_empty_dataset)
