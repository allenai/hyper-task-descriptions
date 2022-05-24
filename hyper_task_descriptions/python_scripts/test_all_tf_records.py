'''
merge tfrecords.
testing this out.
'''
import tensorflow as tf
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool


def test_tf_record(dir):
    for split in ['train', 'test', 'validation']:
        tf_record_list = glob.glob(f'{dir}/{split}.tfrecord*')
        assert len(tf_record_list) < 2, f'{split} has more than one tfrecord file'
        if len(tf_record_list) == 0:
            continue
        dataset = tf.data.TFRecordDataset(tf_record_list)
        try:
            for _ in tqdm(dataset):
                pass
        except tf.python.framework.errors_impl.DataLossError as e:
            print(f'\n============== {dir} {split} failed ==============\n')
            raise e


def main():
    dirs = os.listdir('t0_data_new')
    dirs = [f't0_data_new/{dir}' for dir in dirs]
    pool = Pool()
    for _ in tqdm(pool.imap_unordered(test_tf_record, dirs), total=len(dirs)):
        pass


if __name__ == '__main__':
    main()
