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


def stratified_kfold(n_splits=3, seed=None):
    """
    Returns `StratifiedKFold(n_splits, shuffle=False, random_state=seed)` in a
    way that is compatible with older sklearn versions.

    Please, prefer this method to calling `StratifiedKFold` directly.
    """
    sk_version = _get_sklearn_version()
    if sk_version >= (0, 19):
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(
            n_splits=n_splits, shuffle=False, random_state=seed)
    elif sk_version >= (0, 14) and sk_version < (0, 15):
        # TODO: I know this works for 0.14.1, but perhaps it also works for
        # some newer versions...
        from sklearn.cross_validation import StratifiedKFold
        # sklearn uses numpy's random number generator under the hood, so to
        # get deterministic behavior we have to re-seed it.
        import numpy.random
        class _StratifiedKFold(StratifiedKFold):
            def __init__(self, n_splits, random_state=None):
                self.n_splits = n_splits
                if random_state is not None:
                    numpy.random.seed(seed)
            def split(self, x, y):
                super().__init__(y, n_folds=self.n_splits, indices=True)
                return self
                
        return _StratifiedKFold(n_splits, random_state=seed)

