import os
import csv
import json
from itertools import chain
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.__str__()
        return json.JSONEncoder.default(self, obj)


def get_file_path(file_dir, path_list=[]):
    if os.path.isfile(file_dir):
        path_list.append(file_dir)
        return path_list
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        if os.path.isdir(file_path):
            get_file_path(file_path, path_list)
        else:
            path_list.append(file_path)
    return path_list


def load_labels(label_file):
    labels_dict = {}
    with open(label_file) as f:
        for line in f:
            key, value = line.rstrip('\n').split(':')
            labels_dict[key] = value
    return labels_dict


def per_image_standardization(img):
    mean = np.mean(img)
    std = np.std(img)
    count = 1
    for size in img.shape:
        count = count * size
    adj_std = max(std, 1.0 / np.sqrt(count))
    img = (img - mean) / adj_std
    return img


def load_image(image_file, image_size):
    np_img = cv2.imread(image_file, flags=1)
    np_img = np_img.astype(np.float32)
    np_img = cv2.resize(np_img, (image_size, image_size))
    np_img = per_image_standardization(np_img)
    np_img = np_img[np.newaxis, :, :, [2, 1, 0]]  # BGR2RGB
    # return np_img.ravel()
    return np_img


def convert_to_list(dict_val):
    list_key = list(dict_val)
    list_val = list(dict_val.values())
    list_result = list(chain.from_iterable([zip(list_key, list_val)]))
    return list_result


def output_to_csv(filename, first_row, other_rows):
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(first_row)
        for row in other_rows:
            csv_writer.writerow(row)


def output_to_file(contents, filename):
    if isinstance(contents, dict):
        contents = json.dumps(contents, cls=ArrayEncoder)
        with open(filename, 'w') as f:
            f.write(contents)
    elif isinstance(contents, list):
        with open(filename, 'w') as f:
            for v in contents:
                f.write(v)
    elif isinstance(contents, str):
        with open(filename, 'w') as f:
            f.write(contents)


def to_json_file(info):
    with open("a.json", "w") as f:
        f.write(json.dumps(info, ensure_ascii=False,
                indent=4, separators=(',', ':')))
    return f


def format_percentage(a, b):
    p = 100 * a / b
    if p == 0.0:
        q = '0%'
    else:
        q = '%.2f%%' % p
    return q


def get_pb_node_shape(pb_path, model_node_name):
    tf.reset_default_graph()
    with tf.Session():
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        graph = tf.get_default_graph()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
            selected = [op for op in graph.get_operations() if
                        op.name == model_node_name]
            if selected:
                return selected[0].values()[0].shape
            else:
                print("Not find input_model_node:{}, not get node shape".format(
                    model_node_name))
                return None
