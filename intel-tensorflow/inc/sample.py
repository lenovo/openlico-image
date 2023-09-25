# Copyright 2015-present Lenovo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import csv
import json
from itertools import chain
import cv2


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


def get_pb_node_shape(pb_file, model_node_name):
    tf.reset_default_graph()
    with tf.Session():
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        graph = tf.get_default_graph()
        with open(pb_file, "rb") as f:
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


class Inference():
    def __init__(self, image_path, model_name,
                 output_path, pb_file, label_file):
        self.pb_file = pb_file
        self.config = tf.ConfigProto()
        self.model_name = model_name
        self.label_file = label_file
        self.image_path = image_path
        self.image_size = int(get_pb_node_shape(
            self.pb_file, 'lico_input_node')[1])
        self.labels_dict = load_labels(self.label_file)
        self.output_path = output_path
        self.result_file = 'predict_result'
        # self.config.gpu_options.allow_growth = True
        self.init_model()

    def init_model(self):
        tf.reset_default_graph()
        self.output_graph_def = tf.GraphDef()
        with open(self.pb_file, 'rb') as f:
            self.output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(
                self.output_graph_def,
                input_map=None,
                return_elements=None,
                name=None,
                op_dict=None,
                producer_op_list=None
            )
        self.sess = tf.Session(config=self.config)
        self.input = self.sess.graph.get_tensor_by_name(
            "import/lico_input_node:0")
        self.output = self.sess.graph.get_tensor_by_name(
            "import/lico_output_node:0")

    def predict(self, image_file):
        img = load_image(image_file, self.image_size)
        out = self.sess.run(self.output, feed_dict={self.input: img})
        class_id_val = np.argmax(out)
        return image_file, str(class_id_val)

    def batch_predict(self, image_file_list):
        predict_dict = {}
        start_time = time.time()
        for image_file in image_file_list:
            image_file, class_id = self.predict(image_file)
            predict_dict[image_file] = self.labels_dict[class_id]
            print("[INFO] inference image:",
                  "{} , {}".format(image_file, self.labels_dict[class_id])
                  )
        end_time = time.time()
        print("[INFO] Number of batch pictures: {}\t".format(
            len(image_file_list)),
            "Total time used: {:.2f} sec\t".format(
            end_time - start_time),
            "Sec/example: {:.2f} sec".format(
            (end_time - start_time) / len(image_file_list))
        )
        return predict_dict

    def to_result_file(self, predict_dict):
        output_to_csv(
            os.path.join(self.output_path, self.result_file + '.csv'),
            ['image_path', 'predict_lable'],
            convert_to_list(predict_dict)
        )
        print(f"[INFO] Finished. You can check inference result file: \n"
              f"{os.path.join(self.output_path, self.result_file + '.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python sample.py")
    parser.add_argument("-i", "--image_path", required=True,
                        help="Input image path to do inference.")
    parser.add_argument("-p", "--pb_file",
                        help="inc optimized pb model file, default search "
                             "from the current dir.")
    parser.add_argument("-l", "--label_file",
                        help="For inferencing to real label, default search "
                             "from the current dir.")
    parser.add_argument("-o", "--output_path", default='.',
                        help="Output directory")
    args = parser.parse_args()

    print(f"[INFO] Params: \n"
          f"image_path: {args.image_path} \n"
          f"pb_file: {args.pb_file} \n"
          f"label_file: {args.label_file} \n"
          f"output_path: {args.output_path}")

    # search pb_file & label_file from the parent dir if not input
    parent_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir))
    model_define_file = os.path.join(parent_dir, 'model_define.json')
    if not os.path.isfile(model_define_file):
        raise FileNotFoundError(f'{model_define_file} was not found.')
    with open(model_define_file, 'r') as f:
        model_json = json.loads(f.read())
    model_name = model_json.get('model_name')

    if args.label_file:
        label_file = args.label_file
    else:
        label_file = os.path.join(parent_dir, 'labels.txt')
        if not os.path.isfile(label_file):
            raise FileNotFoundError(f'{label_file} was not found.')
    if args.pb_file:
        pb_file = args.pb_file
    else:
        pb_file = os.path.join(parent_dir, 'inc_optimized_model.pb')
        if not os.path.isfile(pb_file):
            raise FileNotFoundError(f'{pb_file} was not found.')

    inference = Inference(args.image_path, model_name,
                          args.output_path, pb_file, label_file)
    image_file_list = get_file_path(args.image_path)
    predict_dict = inference.batch_predict(image_file_list)
    inference.to_result_file(predict_dict)
