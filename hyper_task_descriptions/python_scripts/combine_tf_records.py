'''
merge tfrecords.
testing this out.
'''
import tensorflow as tf
import glob
import json
import os
from tqdm import tqdm
from multiprocessing import Pool


def merge_tf_records(tf_record_list, output_tf_record):
    dataset = tf.data.TFRecordDataset(tf_record_list)
    # Save dataset to .tfrecord file
    filename = output_tf_record
    with tf.data.experimental.TFRecordWriter(filename) as writer:
        writer.write(dataset)


def transform_dir(dir, new_dir):
    for split in ['train', 'test', 'validation']:
        tf_record_list = glob.glob(f'{dir}/{split}.tfrecord*')
        if len(tf_record_list) == 0:
            continue
        merge_tf_records(tf_record_list, f'{new_dir}/{split}.tfrecord-00000-of-00001')
        # stats and info files.
        if os.path.isfile(f'{dir}/info.{split}.json'):
          with open(f'{dir}/info.{split}.json', 'r') as f:
            data_info = json.load(f)
          data_info['num_shards'] = 1
          with open(f'{new_dir}/info.{split}.json', 'w') as f:
            json.dump(data_info, f)
        if os.path.isfile(f'{dir}/stats.{split}.json'):
          with open(f'{dir}/stats.{split}.json', 'r') as f:
            stats_info = json.load(f)
          with open(f'{new_dir}/stats.{split}.json', 'w') as f:
            json.dump(stats_info, f)

def process_stuff(x):
    old_dir, new_dir, task = x
    if os.path.isfile('t0_data_new/{task}/stats.train.json'):
        return
    #if os.path.isdir(f't0_data_new/{task}'):
    #    return # we already processed this
    #os.mkdir(f't0_data_new/{task}')
    transform_dir(old_dir, new_dir)

def main():
    dirs = os.listdir('t0_data')
    old_dirs = [f't0_data/{dir}' for dir in dirs]
    new_dirs = [f't0_data_new/{dir}' for dir in dirs]

    pool = Pool()
    for _ in tqdm(pool.imap_unordered(process_stuff, zip(old_dirs, new_dirs, dirs)), total=len(dirs)):
        pass


if __name__ == '__main__':
    main()