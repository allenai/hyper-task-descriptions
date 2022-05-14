"""
Point this at a bucket folder with results as formatted by t5x to get min/max/avg scores per prompt.
TODO: also output per-prompt scores in easy-to-paste format.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from google.cloud import storage
from tqdm import tqdm

client = storage.Client(project="ai2-tpu")
bucket = client.bucket("hamishi-tpu-bucket")


# function from:
# https://github.com/bigscience-workshop/architecture-objective/blob/main/bigscience/eval-spreadsheet/parse_promptsource.py
def process_task_prompt(task_prompt: str) -> Tuple[str, str]:
    task_prompt = task_prompt[:-11]  # Remove 'score_eval' string at the end

    task, prompt = None, None
    if "anli" in task_prompt:
        task = "anli" + task_prompt[-3:]
        prompt = task_prompt[5:-3]
    elif "hellaswag" in task_prompt:
        task = "hellaswag"
        prompt = task_prompt[10:]
    elif "story_cloze" in task_prompt:
        task = "story_cloze"
        prompt = task_prompt[17:]
    elif "super_glue" in task_prompt:
        if "cb" in task_prompt:
            task = "cb"
            prompt = task_prompt[14:]
        elif "copa" in task_prompt:
            task = "copa"
            prompt = task_prompt[16:]
        elif "rte" in task_prompt:
            task = "rte"
            prompt = task_prompt[15:]
        elif "wic" in task_prompt:
            task = "wic"
            prompt = task_prompt[15:]
        elif "wsc" in task_prompt:
            task = "wsc"
            prompt = task_prompt[15:]
    elif "winogrande" in task_prompt:
        task = "winogrande"
        prompt = task_prompt[25:]

    if task is None or prompt is None:
        raise ValueError(f"Failed to parse task/prompt: {task_prompt}")

    return task, prompt


def process_ps_results(folder: str) -> Dict[str, Dict[str, float]]:
    accuracies: Dict[str, Dict[str, float]] = defaultdict(dict)
    for blob in tqdm(bucket.list_blobs(prefix=f"{folder}")):
        if blob.name.endswith("-metrics.jsonl"):
            s = blob.download_as_string().decode("utf-8")
            filename = Path(blob.name).stem.replace("-metrics", "")
            task, prompt = process_task_prompt(filename)
            # last step
            accuracies[task][prompt] = json.loads(s.split("\n")[-1])["accuracy"]
    return accuracies


# min, max, average per prompt
def summarise_ps_results(accuracies: Dict[str, Dict[str, float]]) -> None:
    print("TASK: MIN MAX AVG")
    for task in accuracies:
        scores = [x for x in accuracies[task].values()]
        print(f"{task}: {min(scores):.2f} {max(scores):.2f} {sum(scores) / len(scores):.2f}")
    print("-------------" * 2)


def print_all_results(accuracies: Dict[str, Dict[str, float]]) -> None:
    for task in accuracies:
        for prompt, score in accuracies[task].items():
            print(f"{task} {prompt} {score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get formatted results from p3/t0 eval mixture")
    parser.add_argument(
        "-f",
        "--res-folder",
        type=str,
        help="Path to a promptsource .json result file",
    )
    args = parser.parse_args()

    accuracies = process_ps_results(args.res_folder)
    summarise_ps_results(accuracies)
    print_all_results(accuracies)
    print("You should be able to copy/paste the above! :)")
