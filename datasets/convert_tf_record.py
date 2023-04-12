import math
import os
import random
import sys

import tensorflow as tf
import datasets.dataset_utils as dataset_utils

RANDOM_SEED = 0



class ImageReader(object):

    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)


    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data);
        return image.shape[0], image.shape[1]



    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image



def _get_dataset_filename(tfrecord_dir, split_name, shard_id, dataset_name, num_shards):
    output_filename = dataset_name + '_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(tfrecord_dir, output_filename)




def _dataset_exists(tf_record_dir, dataset_name, num_shards):
    for split_name in ['train', 'eval']:
        for shard_id in range(num_shards):
            output_filename = _get_dataset_filename(tf_record_dir, split_name, shard_id, dataset_name, num_shards)
            if not tf.gfile.Exists(output_filename):
                return False

    return True





def _get_filenames_and_classes(dataset_dir):
    image_root = os.path.join(dataset_dir, "images")
    directories = []
    class_names = []

    for filename in os.listdir(image_root):
        path = os.path.join(image_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames_dict = {}
    for idx, directory in enumerate(directories):
        photo_filenames_dict[idx] = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames_dict[idx].append(path)

    return photo_filenames_dict, sorted(class_names)




def _convert_dataset(split_name, filenames, class_names_to_idx, dataset_dir, dataset_name, tf_record_dir, num_shards):
    assert split_name in ['train', 'eval']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session() as sess:
            for shard_id in range(num_shards):
                output_filename = _get_dataset_filename(tf_record_dir, split_name, shard_id, dataset_name, num_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>>Converting [%s] image %d/%d shard %d' % (split_name, i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_idx[class_name]

                        filename = os.path.basename(filenames[i])
                        sample = dataset_utils.image_to_tfexample(image_data, os.path.join(class_name, filename), b'jpg', height, width, class_id)
                        tfrecord_writer.write(sample.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()



def run(dataset_name, dataset_dir, num_shards, ratio_eval):
    tf_record_dir = os.path.join(dataset_dir, 'TFRecord')
    if not tf.gfile.Exists(tf_record_dir):
        tf.gfile.MakeDirs(tf_record_dir)

    if _dataset_exists(tf_record_dir, dataset_name, num_shards):
        print('TFRecord files already exist. Exiting witout re-creating them.')
        return

    
    photo_filenames_dict, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_idx = dict(zip(class_names, range(len(class_names))))
    
    random.seed(RANDOM_SEED)

    training_filenames = []
    evaluation_filenames = []
    for photo_filenames in photo_filenames_dict.values():
        random.shuffle(photo_filenames)
        num_evaluation = int(len(photo_filenames) * ratio_eval)
        training_filenames.extend(photo_filenames[num_evaluation:])
        evaluation_filenames.extend(photo_filenames[:num_evaluation])

    random.shuffle(training_filenames)
    random.shuffle(evaluation_filenames)

    _convert_dataset('train', training_filenames, class_names_to_idx, dataset_dir, dataset_name, tf_record_dir, num_shards)
    _convert_dataset('eval', evaluation_filenames, class_names_to_idx, dataset_dir, dataset_name, tf_record_dir, num_shards)


    idx_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(idx_to_class_names, tf_record_dir)

    print('\nFinished converting the dataset!')

