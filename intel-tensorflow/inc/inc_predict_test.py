import json
import os
import sys
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from inc_common import load_labels, load_image, get_pb_node_shape


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
        tf.Graph().as_default()
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

    def predict(self):
        start_time = time.time()
        img = load_image(self.image_path, self.image_size)
        out = self.sess.run(self.output, feed_dict={self.input: img})
        score = tf.nn.softmax(out, name='pre')
        score_val = tf.keras.backend.eval(score)
        class_id = tf.math.argmax(score, axis=1)
        class_id_val = tf.keras.backend.eval(class_id[0])
        label = self.labels_dict[str(class_id_val)]
        data = [{'label': label, 'probability': float(np.max(score_val[0]))}]
        end_time = time.time()
        duration = int(round(1000 * (end_time - start_time)))

        return {'data': data, 'duration': duration}


def do_inference(scenario, model_path, model_name,
                 image_file, output_dir, **kwargs):
    print("Deal with file {}".format(image_file))
    pb_model_path = model_path + "/inc_optimized_model.pb"
    label_path = model_path + "/labels.txt"
    if scenario == "classification":
        try:
            inference = Inference(image_file, model_name,
                                  output_dir, pb_model_path, label_path)
            inf_result = inference.predict()
            temp_file = os.path.join(
                output_dir, 'temp_file_{}.json'.format(int(time.time()))
            )
            with open(temp_file, 'w') as f:
                inf_result['filename'] = image_file
                f.write(json.dumps(inf_result, indent=4))
                print("{0}".format(inf_result))
            response_file = os.path.join(output_dir, 'result.json')
            os.rename(temp_file, response_file)
            print("Save inference result to {}".format(response_file))
        except Exception as e:
            error_file = os.path.join(output_dir, 'error.json')
            os.mknod(error_file)
            raise e
        sys.stdout.flush()
    else:
        pass


class Inference_server():
    def __init__(self, model_name, pb_path, label_path):
        self.pb_path = pb_path
        self.config = tf.ConfigProto()
        self.model_name = model_name
        self.label_file = label_path
        self.image_size = int(get_pb_node_shape(
            self.pb_path, 'lico_input_node')[1])
        self.labels_dict = load_labels(self.label_file)
        self.result_file = 'predict_result'
        # self.config.gpu_options.allow_growth = True
        self.init_model()

    def init_model(self):
        tf.Graph().as_default()
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

    def predict(self, img_path):
        start_time = time.time()
        img = load_image(img_path, self.image_size)
        out = self.sess.run(self.output, feed_dict={self.input: img})
        score = tf.nn.softmax(out, name='pre')
        score_val = tf.keras.backend.eval(score)
        class_id = tf.math.argmax(score, axis=1)
        class_id_val = tf.keras.backend.eval(class_id[0])
        label = self.labels_dict[str(class_id_val)]
        data = [{'label': label, 'probability': float(np.max(score_val[0]))}]
        end_time = time.time()
        duration = int(round(1000 * (end_time - start_time)))

        return {'data': data, 'duration': duration}


if __name__ == '__main__':
    tf.app.flags.DEFINE_enum(
        'scenario', 'classification',
        ['classification', 'objectdetection', 'segmentation', 'text'],
        "Scenario for this service"
    )
    tf.app.flags.DEFINE_string(
        'lico_model_path', '', "model path, for inference job"
    )
    tf.app.flags.DEFINE_string(
        'lico_image_file', '', "image file to deal, for inference job"
    )
    tf.app.flags.DEFINE_string(
        'lico_output_dir', '', "output dir, for inference job"
    )
    tf.app.flags.DEFINE_string('lico_model_name', '',
                               "model_name for inference job")
    tf.app.flags.DEFINE_boolean('lico_text_lower_case', True,
                                "text for do_lower_case")
    tf.app.flags.DEFINE_enum('lico_text_language', 'en',
                             ['en', 'zh'], 'text for language')
    tf.app.flags.DEFINE_string("lico_text_content", "", "text content")
    tf.app.flags.DEFINE_integer('lico_text_max_seq_length', 256,
                                "text for max_seq_length")
    FLAGS = tf.app.flags.FLAGS

    do_inference(
        FLAGS.scenario,
        FLAGS.lico_model_path,
        FLAGS.lico_model_name,
        FLAGS.lico_image_file,
        FLAGS.lico_output_dir
    )
