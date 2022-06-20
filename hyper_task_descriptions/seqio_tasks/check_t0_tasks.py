"""
Quick script for checking if T0 caching is complete.
"""
from google.cloud import storage

storage_client = storage.Client(project="ai2-allennlp")
bucket = storage_client.bucket("hamishi-tpu-bucket")

name = "COMPLETED"

tasks = [t.strip().decode("utf-8") for t in open("all_t0_tasks.txt", "rb").readlines()]

for task in tasks:
    if not storage.Blob(bucket=bucket, name=f"t0_data_split_descr/{task}/COMPLETED").exists(
        storage_client
    ):
        print(task)
