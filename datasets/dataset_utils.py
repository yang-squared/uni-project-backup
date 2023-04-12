import os
import sys


import tensorflow as tf
import tf_slim as slim

LABELS_FILENAME = 'labels.txt'



def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))



def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))




def write_label_file(idx_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in idx_to_class_names:
            class_name = idx_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))



def has_labels(dataset_dir, filename=LABELS_FILENAME):
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))



def image_to_tfexample(image_data, filename, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature= {
        'image/encoded': bytes_feature(image_data),
        'image/format' : bytes_feature(image_format),
        'image/class/label' : int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
    }))



def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()

    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]

    return labels_to_class_names




def load_batch(dataset, batch_size, height, width, num_classes, is_training=True, do_scaling=True, use_grayscale=True, use_standardization=True, use_color_distortion=False, use_crop_distortion=False, crop_ratio=1.0):
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label, filename] = provider.get(['image', 'label', 'filename'])

    final_width = width
    final_height = height

    if use_crop_distortion:
        width = int(float(width) * crop_ratio)
        height = int(float(height) * crop_ratio)


    image = normalize_image(image, height, width, do_scaling, use_standardization)
    """
    if do_scaling:
        new_size = tf.constant([height, width])
        image = tf.image.resize_images(image, new_size)

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)


    if use_standardization:
        image = tf.image.per_image_standardization(image)
    """


    if use_crop_distortion:
        image = tf.random_crop(image, [final_height, final_width, 3])

    if use_color_distortion:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)

    if use_grayscale:
        image = tf.image.rgb_to_grayscale(image)
        image.set_shape([final_height, final_width, 1])

    image = tf.image.random_flip_left_right(image)
    image = tf.to_float(image)

    one_hot_labels = slim.one_hot_encoding(label, num_classes)

    images, labels, filenames = tf.train.batch(
        [image, one_hot_labels, filename],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    #tf.summary.text("filename", filenames)
    return images, labels, filenames, dataset.num_samples

    



def normalize_image(image, height, width, do_scaling=True, use_standardization=True):
    if do_scaling:
        new_size = tf.constant([height, width])
        image = tf.image.resize_images(image, new_size)

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)

    if use_standardization:
        image = tf.image.per_image_standardization(image)

    return image





def load_image(image, height, width, do_scaling=True, use_standardization=True, use_grayscale=True):
    image = normalize_image(image, height, width, do_scaling, use_standardization)

    output_channels = 1 if use_grayscale == True else 3
    if use_grayscale:
        image = tf.image.rgb_to_grayscale(image)
        image.set_shape([height, width, 1])

    image = tf.reshape(image, [1, height, width, output_channels])
    return tf.to_float(image)