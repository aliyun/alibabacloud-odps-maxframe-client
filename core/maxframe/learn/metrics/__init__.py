# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from maxframe.learn.metrics import _check_targets
from maxframe.learn.metrics._classification import (
    accuracy_score,
    f1_score,
    fbeta_score,
    log_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from maxframe.learn.metrics._ranking import auc, roc_auc_score, roc_curve
from maxframe.learn.metrics._regression import r2_score
from maxframe.learn.metrics.pairwise import pairwise_distances

# isort: off
from maxframe.learn.metrics._scorer import get_scorer
