"""
merge tfrecords.
testing this out.
"""
import glob
import json
import os

import tensorflow as tf
from tqdm import tqdm


def merge_tf_records(tf_record_list, output_tf_record):
    dataset = tf.data.TFRecordDataset(tf_record_list)
    # Save dataset to .tfrecord file
    filename = output_tf_record
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)


def transform_dir(dir, new_dir):
    for split in ["train", "test", "validation"]:
        tf_record_list = glob.glob(f"{dir}/{split}.tfrecord*")
        if len(tf_record_list) == 0:
            continue
        merge_tf_records(tf_record_list, f"{new_dir}/{split}.tfrecord-00000-of-00001")
        # stats and info files.
        with open(f"{dir}/info.{split}.json", "r") as f:
            data_info = json.load(f)
        data_info["num_shards"] = 1
        with open(f"{new_dir}/info.{split}.json", "w") as f:
            json.dump(data_info, f)

        with open(f"{dir}/stats.{split}.json", "r") as f:
            stats_info = json.load(f)
        with open(f"{new_dir}/stats.{split}.json", "w") as f:
            json.dump(stats_info, f)


def main():
    dirs = os.listdir("t0_data")
    old_dirs = [f"t0_data/{dir}" for dir in dirs]
    new_dirs = [f"t0_data_new/{dir}" for dir in dirs]
    for old_dir, new_dir in tqdm(zip(old_dirs, new_dirs)):
        transform_dir(old_dir, new_dir)


if __name__ == "__main__":
    main()
