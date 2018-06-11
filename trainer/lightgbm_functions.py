# Copyright 2018 Aloisio Dourado
# Copyright 2018 Sophie Arana
# Copyright 2018 Johanna de Vos
# Copyright 2018 Tom Westerhout
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
from builtins import int, super
import logging


class StoreLoggingLevel(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('`nargs` is not supported.')
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        level = getattr(logging, value.upper(), None)
        if not isinstance(level, int):
            raise ValueError('Invalid log level: {}'.format(value))
        setattr(namespace, self.dest, level)


def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file', help='Path to training data', required=True)
    parser.add_argument(
      '--valid-file', help='Path to validation data', required=False)
    parser.add_argument(
      '--test-file', help='Path to test data', required=False)
    parser.add_argument(
      '--number-lines', help='Number of lines to use.', required=False)
    parser.add_argument(
        '--job-dir',
        help='Directory where to store checkpoints and exported models',
        default='.')
    parser.add_argument(
      '--run', 
      help='Choose <optimization> to run the cross-validation, or \
      <submission> to train on all the training data',
      choices = ['plot','optimization', 'submission'],
      required=True)
    parser.add_argument(
        '--log', help='Logging level', default=logging.DEBUG,
        action=StoreLoggingLevel)
    return parser
