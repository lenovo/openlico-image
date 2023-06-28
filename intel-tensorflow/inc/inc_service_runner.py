# Copyright 2015-2023 Lenovo
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

import base64
import sys
import traceback
from tempfile import NamedTemporaryFile

import falcon
from falcon.media.validators import jsonschema

from inc_predict_test import Inference_server


def get_infe_serv(model_path, model_name):
    print('Init Load Model')
    pb_model_path = model_path + "/inc_optimized_model.pb"
    label_path = model_path + "/labels.txt"
    inference_service_mgt = Inference_server(
        model_name, pb_model_path, label_path)
    print('Load Model Success')
    return inference_service_mgt


class IncServiceRunner(object):
    def __init__(self, model_path, model_name):
        self.InferenceServiceMgt = get_infe_serv(
            model_path, model_name
        )
        sys.stdout.flush()

    @jsonschema.validate(
        {
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "image_type": {
                    "type": "string",
                    "enum": ["BASE64"]
                },
            },
            "required": ["image", "image_type"]
        }
    )
    def on_post(self, req, resp):
        try:
            content = base64.b64decode(req.media.get('image'))
            with NamedTemporaryFile(mode='wb') as buf:
                buf.write(content)
                buf.flush()
                inf_result = self.InferenceServiceMgt.predict(buf.name)
        except ValueError:
            traceback.print_exc()
            raise falcon.HTTPBadRequest(
                description='The format of image is unsupported'
            )
        except TypeError:
            traceback.print_exc()
            raise falcon.HTTPBadRequest(
                description='It is not an image'
            )
        except Exception:
            traceback.print_exc()
            raise falcon.HTTP_INTERNAL_SERVER_ERROR(
                description='Internal error of inference server'
            )
        resp.media = inf_result
