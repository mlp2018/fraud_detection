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
from builtins import (bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)
import sklearn


def _get_sklearn_version():
    return tuple(map(int, sklearn.__version__.split('.')))


def stratified_kfold(x, y, n_splits=3, seed=None):
    """
    Returns 
    `StratifiedKFold(n_splits, shuffle=False, random_state=seed).split(x, y)`
    in a way compatible with older sklearn versions.

    Please, use this method to calling `StratifiedKFold` directly.
    """
    sk_version = _get_sklearn_version()
    if sk_version >= (0, 19):
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(
            n_splits=n_splits, shuffle=False, random_state=seed).split(x, y)
    elif sk_version >= (0, 14) and sk_version < (0, 15):
        # TODO: I know this works for 0.14.1, but perhaps it also works for
        # some newer versions...
        from sklearn.cross_validation import StratifiedKFold
        # sklearn uses numpy's random number generator under the hood, so to
        # get deterministic behavior we have to re-seed it.
        import numpy
        import numpy.random
        numpy.random.seed(seed)  
        return StratifiedKFold(y, n_folds=n_splits, indices=True)

