"""
A really basic, easy to learn task involving numeric operators.
Using to test the model setups
"""
import random
from operator import add, sub

import seqio
import tensorflow as tf

from hyper_task_descriptions.hf_vocab import HuggingfaceVocabulary

TaskRegistry = seqio.TaskRegistry

t5_vocab = HuggingfaceVocabulary("t5-base")
roberta_vocab = HuggingfaceVocabulary("roberta-base", add_special_tokens=True)


# copied from the linux words dict
words = [
    line.strip() for line in open("hyper_task_descriptions/numeric_task/words.txt", "r").readlines()
]


NUMERIC_FEATURES = {
    "inputs": "10 10",
    "hyper_inputs": "multiply the two numbers",
    "targets": "100",
    "task_names": [1],
}

# INITIAL COPY TASK ##


def get_copy_data():
    numbers = list(range(1, 10))
    operators = ["+", "-"]
    operator_fns = [add, sub]
    for n1 in numbers:
        for n2 in numbers[n1:]:
            for i, (o, fn) in enumerate(zip(operators, operator_fns)):
                yield {
                    "inputs": f"{n1} {n2}",
                    "hyper_inputs": f"{fn(n1, n2)}",
                    "targets": f"{fn(n1, n2)}",  # answers[0] if fn(n1, n2) > 0 else answers[1],
                    "task_names": [fn(n1, n2)],
                }


def construct_copy_dataset(split, shuffle_files, seed):
    return tf.data.Dataset.from_generator(
        get_copy_data,
        output_signature={k: tf.type_spec_from_value(v) for k, v in NUMERIC_FEATURES.items()},
    )


seqio.TaskRegistry.add(
    "copy_task",
    source=seqio.FunctionDataSource(construct_copy_dataset, splits=["train", "test"]),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
        seqio.CacheDatasetPlaceholder(required=False),
    ],
    output_features={
        "inputs": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "targets": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "hyper_inputs": seqio.Feature(vocabulary=t5_vocab, required=False, add_eos=True),
        "task_names": seqio.Feature(seqio.PassThroughVocabulary(1), add_eos=False, dtype=tf.int32),
    },
    metric_fns=[],
)

# MATH FN TASK, WITHOUT SWAPPING LABELS


def get_math_data():
    numbers = list(range(1, 10))
    operators = ["+", "-"]  # , "*", "//"]
    operator_fns = [add, sub]  # , mul, floordiv]
    for n1 in numbers:
        for n2 in numbers[n1:]:
            for i, (o, fn) in enumerate(zip(operators, operator_fns)):
                yield {
                    "inputs": f"{n1} {n2}",
                    "hyper_inputs": f"{o}",
                    "targets": f"{fn(n1, n2)}",
                    "task_names": [fn(n1, n2)],
                }


def construct_math_dataset(split, shuffle_files, seed):
    return tf.data.Dataset.from_generator(
        get_math_data,
        output_signature={k: tf.type_spec_from_value(v) for k, v in NUMERIC_FEATURES.items()},
    )


seqio.TaskRegistry.add(
    "math_fn_task",
    source=seqio.FunctionDataSource(construct_math_dataset, splits=["train", "test"]),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
        seqio.CacheDatasetPlaceholder(required=False),
    ],
    output_features={
        "inputs": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "targets": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "hyper_inputs": seqio.Feature(vocabulary=t5_vocab, required=False, add_eos=True),
        "task_names": seqio.Feature(seqio.PassThroughVocabulary(1), add_eos=False, dtype=tf.int32),
    },
    metric_fns=[],
)

# MATH FN TASK, WITHOUT SWAPPING LABELS


def get_math_label_data():
    numbers = list(range(1, 10))
    operators = ["+", "-"]  # , "*", "//"]
    operator_fns = [add, sub]  # , mul, floordiv]
    for n1 in numbers:
        for n2 in numbers[n1:]:
            for i, (o, fn) in enumerate(zip(operators, operator_fns)):
                word_a = random.choice(words)
                word_b = random.choice(words)
                yield {
                    "inputs": f"{n1} {n2}",
                    "hyper_inputs": f"Perform {o} on the two inputs. Output {word_a} if the answer"
                    f"is even and {word_b} otherwise.",
                    "targets": f"{word_a if fn(n1, n2) % 2 == 0 else word_b}",
                    "task_names": [fn(n1, n2)],
                }


def construct_math_label_dataset(split, shuffle_files, seed):
    return tf.data.Dataset.from_generator(
        get_math_label_data,
        output_signature={k: tf.type_spec_from_value(v) for k, v in NUMERIC_FEATURES.items()},
    )


seqio.TaskRegistry.add(
    "math_label_task",
    source=seqio.FunctionDataSource(construct_math_label_dataset, splits=["train", "test"]),
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
        seqio.CacheDatasetPlaceholder(required=False),
    ],
    output_features={
        "inputs": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "targets": seqio.Feature(vocabulary=t5_vocab, add_eos=True),
        "hyper_inputs": seqio.Feature(vocabulary=t5_vocab, required=False, add_eos=True),
        "task_names": seqio.Feature(seqio.PassThroughVocabulary(1), add_eos=False, dtype=tf.int32),
    },
    metric_fns=[],
)
