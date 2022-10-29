import functools
import random
import re
import string

import seqio
import tensorflow as tf
import transformers
from datasets import load_dataset

from hyper_task_descriptions.hf_vocab import HuggingfaceVocabulary
from hyper_task_descriptions.ni_tasks.evaluation import compute_metrics
from hyper_task_descriptions.ni_tasks.ni_collator import DataCollatorForNI
from hyper_task_descriptions.seqio_tasks.utils import hf_dataset_to_tf_dataset

seqio.add_global_cache_dirs(["gs://hamishi-us-bucket/ni_t5_pre_eos_200"])


def get_ni_data(
    split,
    shuffle_files,
    seed,
    max_num_instances_per_task,
    max_num_instances_per_eval_task,
    raw_input,
    alt_raw_input,
    **ni_collator_args
):
    # HF datasets does not support file-level shuffling
    random_gen = random.Random(seed)
    del shuffle_files
    dataset = load_dataset(
        "hyper_task_descriptions/ni_tasks/ni_dataset.py",
        # data_dir="../natural-instructions/",
        max_num_instances_per_task=max_num_instances_per_task,
        max_num_instances_per_eval_task=max_num_instances_per_eval_task,
        split=split,
    )
    dataset = dataset.shuffle(seed=seed)

    # if not raw_input, we will use the following collator to add definition and examples
    # to the input, as we did for Tk-Instruct.
    def input_transform_func(x):
        return x
    def hyper_input_transform_func(x):
        return x
    if not raw_input:
        data_collator = DataCollatorForNI(**ni_collator_args)
    elif alt_raw_input:
        # conversion for regular input
        prefix_string = "Now complete the following example -\n"
        prefix_string += "Input: "
        suffix_string = "\nOutput: "

        def input_transform_func(x):
            x = x.strip()
            if x[-1] not in string.punctuation:
                x += "."
            return prefix_string + x + suffix_string

        def hyper_input_transform_func(x):
            x = x.strip()
            if x[-1] not in string.punctuation:
                x += "."
            return "Definition: " + x + "\n\n"

    def convert_format(example):
        task_idx = re.findall(r"^task(\d+)_", example["Task"])
        assert len(task_idx) == 1
        task_idx = int(task_idx[0])
        return {
            "id": example["id"],
            "inputs": input_transform_func(example["Instance"]["input"])
            if raw_input
            else data_collator([example])["inputs"][0].strip(),
            "hyper_inputs": hyper_input_transform_func(example["Definition"][0]),
            "targets": random_gen.choice(example["Instance"]["output"]),
            "references": example["Instance"]["output"],
            "task_names": tf.constant([task_idx], dtype=tf.int32),
        }

    original_columns = dataset.column_names
    dataset = dataset.map(convert_format)
    # map keeps original columns, remove them
    dataset = dataset.remove_columns(
        set(original_columns)
        - {"id", "inputs", "hyper_inputs", "targets", "references", "task_names"}
    )
    return hf_dataset_to_tf_dataset(dataset)


# use my unique vocab instead.
t5_vocab = HuggingfaceVocabulary("t5-base")

output_features = {
    "inputs": seqio.Feature(t5_vocab, add_eos=True, dtype=tf.int32),
    "hyper_inputs": seqio.Feature(t5_vocab, add_eos=False, dtype=tf.int32),
    "targets": seqio.Feature(t5_vocab, add_eos=True, dtype=tf.int32),
    "task_names": seqio.Feature(seqio.PassThroughVocabulary(1), add_eos=False, dtype=tf.int32),
}

preprocessors = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(required=False),
    seqio.preprocessors.append_eos,
]


def postprocessor(output_or_target, example=None, is_target=False):
    """Returns output as answer, or all answers if the full example is provided."""
    if is_target:
        return [it.decode("utf-8") for it in example["references"]]
    else:
        return output_or_target


def ni_metrics_wrapper(targets, predictions):
    return compute_metrics(predictions=predictions, references=targets, xlingual=False)


dataset_fn = functools.partial(
    get_ni_data,
    seed=None,
    max_num_instances_per_task=100,
    max_num_instances_per_eval_task=100,
    raw_input=True,
)

data_source = seqio.FunctionDataSource(
    dataset_fn,
    splits=["train", "test"],
)

seqio.TaskRegistry.add(
    "natural_instructions",
    data_source,
    preprocessors=preprocessors,
    output_features=output_features,
    postprocess_fn=postprocessor,
    metric_fns=[ni_metrics_wrapper],
    shuffle_buffer_size=50000,  # default of 1000 is too small
)

dataset_fn = functools.partial(
    get_ni_data,
    seed=None,
    max_num_instances_per_task=100,
    max_num_instances_per_eval_task=100,
    raw_input=False,
    tokenizer=transformers.AutoTokenizer.from_pretrained("t5-base"),
    model=None,
    max_source_length=1024,
    max_target_length=512,
    add_task_definition=True,
    num_pos_examples=2,
    num_neg_examples=0,
    add_explanation=False,
    text_only=True,
)

data_source = seqio.FunctionDataSource(
    dataset_fn,
    splits=["train", "test"],
)

seqio.TaskRegistry.add(
    "natural_instructions_def_pos_2",
    data_source,
    preprocessors=preprocessors,
    output_features=output_features,
    postprocess_fn=postprocessor,
    metric_fns=[ni_metrics_wrapper],
    shuffle_buffer_size=50000,  # default of 1000 is too small
)

dataset_fn = functools.partial(
    get_ni_data,
    seed=None,
    max_num_instances_per_task=100,
    max_num_instances_per_eval_task=100,
    raw_input=False,
    tokenizer=transformers.AutoTokenizer.from_pretrained("t5-base"),
    model=None,
    max_source_length=1024,
    max_target_length=512,
    add_task_definition=True,
    num_pos_examples=0,
    num_neg_examples=0,
    add_explanation=False,
    text_only=True,
)

data_source = seqio.FunctionDataSource(
    dataset_fn,
    splits=["train", "test"],
)

seqio.TaskRegistry.add(
    "natural_instructions_def",
    data_source,
    preprocessors=preprocessors,
    output_features=output_features,
    postprocess_fn=postprocessor,
    metric_fns=[ni_metrics_wrapper],
    shuffle_buffer_size=100000,  # default of 1000 is too small
)

dataset_fn = functools.partial(
    get_ni_data,
    seed=None,
    max_num_instances_per_task=100,
    max_num_instances_per_eval_task=100,
    raw_input=True,
    alt_raw_input=True
)

data_source = seqio.FunctionDataSource(
    dataset_fn,
    splits=["train", "test"],
)

seqio.TaskRegistry.add(
    "natural_instructions_hyper_alt",
    data_source,
    preprocessors=preprocessors,
    output_features=output_features,
    postprocess_fn=postprocessor,
    metric_fns=[ni_metrics_wrapper],
    shuffle_buffer_size=50000,  # default of 1000 is too small
)