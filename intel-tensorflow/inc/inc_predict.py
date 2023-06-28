import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from inc_common import (load_labels, output_to_csv, output_to_file,
                        convert_to_list, get_file_path, load_image,
                        format_percentage, get_pb_node_shape)


class Inference():
    def __init__(self, image_path, model_name,
                 result_path, pb_path, label_path):
        self.pb_path = pb_path
        self.config = tf.ConfigProto()
        self.model_name = model_name
        self.label_file = label_path
        self.image_path = image_path
        self.image_size = int(get_pb_node_shape(
            self.pb_path, 'lico_input_node')[1])
        self.labels_dict = load_labels(self.label_file)
        self.result_path = result_path
        self.result_file = 'predict_result'
        # self.config.gpu_options.allow_growth = True
        self.init_model()

    def init_model(self):
        tf.reset_default_graph()
        self.output_graph_def = tf.GraphDef()
        with open(self.pb_path, 'rb') as f:
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
        image_file = image_file.replace(self.image_path.rstrip('/'), '.')
        return image_file, str(class_id_val)

    def batch_predict(self, image_file_list):
        predict_dict = {}
        start_time = time.time()
        true_file = 0
        for image_file in image_file_list:
            image_file, class_id = self.predict(image_file)
            predict_dict[image_file] = self.labels_dict[class_id]
            print("[INFO] inference image:",
                  "{} , {}".format(image_file, self.labels_dict[class_id])
                  )
            if image_file.split('/')[-2] == self.labels_dict[class_id]:
                true_file += 1
        end_time = time.time()
        print("[INFO] Number of batch pictures: {}\t".format(
            len(image_file_list)),
            "Total time used: {:.2f} sec\t".format(
            end_time - start_time),
            "Sec/example: {:.2f} sec".format(
            (end_time - start_time) / len(image_file_list))
        )
        print("[INFO] Accuracy rate:", format_percentage(
            true_file, len(image_file_list)))

        return predict_dict

    def to_result_file(self, predict_dict):
        output_to_csv(
            os.path.join(self.result_path, self.result_file + '.csv'),
            ['image_path', 'predict_lable'],
            convert_to_list(predict_dict)
        )
        output_to_file(
            predict_dict,
            os.path.join(self.result_path, self.result_file + '.txt')
        )
        print(f"[INFO] Finished. You can check inference result file: \n"
              f"{os.path.join(self.result_path, self.result_file + '.csv')} \n"
              f"{os.path.join(self.result_path, self.result_file + '.txt')}")


def main(image_path, model_name, result_path, pb_path, label_path):
    inference = Inference(image_path, model_name,
                          result_path, pb_path, label_path)
    image_file_list = get_file_path(image_path)
    predict_dict = inference.batch_predict(image_file_list)
    inference.to_result_file(predict_dict)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('model_path', '', '')
    tf.app.flags.DEFINE_string('label_file', '', '')
    tf.app.flags.DEFINE_string('model_name', '', '')
    tf.app.flags.DEFINE_string('input_dir', '', '')
    tf.app.flags.DEFINE_string('output_dir', '', '')
    FLAGS = tf.app.flags.FLAGS
    print(f"[INFO] Params: \n"
          f"model_name: {FLAGS.model_name} \n"
          f"model_path: {FLAGS.model_path} \n"
          f"label_file: {FLAGS.label_file} \n"
          f"image_path: {FLAGS.input_dir} \n"
          f"output_path: {FLAGS.output_dir}")
    main(FLAGS.input_dir, FLAGS.model_name, FLAGS.output_dir,
         FLAGS.model_path, FLAGS.label_file)
