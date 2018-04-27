#!/usr/bin/env python

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


from __future__ import absolute_import, division, print_function
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)
from argparse import ArgumentParser
import sklearn
import pandas as pd


def _get_sklearn_version():
    (release, major, minor) = sklearn.__version__.split('.')
    return int(release), int(major), int(minor)


def make_cmd_args_parser():
    parser = ArgumentParser(
        description='Splits our data-file into two parts: '
                    'for training and for validation.')
    parser.add_argument(
        'input_file', help='File to be split.')
    parser.add_argument(
        'training_file', help='Where to store the training data.')
    parser.add_argument(
        'validation_file', help='Where to store the validation data.')
    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='Fraction of data to use for validation.')
    parser.add_argument(
        '--seed', type=int, default=None, help='Random seed to use.')
    return parser


def split(x, y, alpha, seed):
    if _get_sklearn_version() >= (0, 19, 0):
        from sklearn.model_selection import StratifiedShuffleSplit
        return next(iter(StratifiedShuffleSplit(
            n_splits=1, test_size=alpha,
            random_state=seed).split(x, y)))
    else:  # We have an old version of sklearn...
        from sklearn.cross_validation import StratifiedShuffleSplit
        return next(iter(StratifiedShuffleSplit(
            y, n_iter=1, test_size=alpha,
            random_state=seed)))


def main():
    args = make_cmd_args_parser().parse_args()
    whole_dataset = pd.read_csv(args.input_file)
    train_indices, test_indices = split(
        whole_dataset, whole_dataset['is_attributed'], args.alpha, args.seed)
    whole_dataset.take(train_indices, is_copy=False).to_csv(
        args.training_file, index=False)
    whole_dataset.take(test_indices, is_copy=False).to_csv(
        args.validation_file, index=False)


if __name__ == '__main__':
    main()
