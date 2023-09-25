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
from subprocess import check_call

from inc_service_runner import IncServiceRunner


def create_app():
    import falcon

    api = falcon.API()

    api.req_options.strip_url_path_trailing_slash = True

    model_name = os.environ.get('LETRAIN_MODEL_NAME', '')
    model_path = os.environ.get('LICO_MODEL_PATH', '')

    service_url = os.environ.get('SERVICE_URL', '')
    api.add_route(
        service_url,
        IncServiceRunner(model_path, model_name)
    )

    return api


if __name__ == 'inc_service':
    inf_app = create_app()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default='/api/service')
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("-n", "--model_name", required=True,
                        help="Original model topology.")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Original model topology.")
    args = parser.parse_args()

    cmd = [
        'gunicorn',
        '--workers',
        '1',
        '--bind',
        '0.0.0.0:{0}'.format(args.port),
        '--pythonpath',
        os.path.dirname(__file__),
        '--access-logfile',
        '-',
        '--error-logfile',
        '-',
        '--timeout',
        '600',
        '--env',
        'LETRAIN_MODEL_NAME={0}'.format(args.model_name),
        '--env',
        'LICO_MODEL_PATH={0}'.format(args.model_path),
        '--env',
        'SERVICE_URL={0}'.format(args.url),
        'inc_service:inf_app'
    ]
    check_call(cmd)
