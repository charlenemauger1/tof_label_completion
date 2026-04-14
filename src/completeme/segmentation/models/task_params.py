# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

task_name = {
    "11": "Dataset011_CAP_SAX",
    "12": "Dataset012_CAP_2CH",
    "13": "Dataset013_CAP_3CH",
    "14": "Dataset014_CAP_4CH",
    "15": "Dataset015_CAP_RVOT",
    "16": "Dataset016_CAP_RVT",
}

patch_size = {
    "11": [224, 256],
    "12": [256, 256],
    "13": [256, 224],
    "14": [256, 224],
    "15": [224, 256],
    "16": [256, 256],
}

spacing = {
    "11": [1.328125, 1.328125],
    "12": [1.3672003746032715, 1.3671998977661133],
    "13": [1.328125, 1.328125],
    "14": [1.328125, 1.328125],
    "15": [1.3333333730697632, 1.3333333730697632],
    "16": [1.40625, 1.40625],
}

clip_values = {
    "11": [0, 0],
    "12": [0, 0],
    "13": [0, 0],
    "14": [0, 0],
    "15": [0, 0],
    "16": [0, 0],
}

normalize_values = {
    "11": [0, 0],
    "12": [0, 0],
    "13": [0, 0],
    "14": [0, 0],
    "15": [0, 0],
    "16": [0, 0],
}

data_loader_params = {
    "11": {"batch_size": 1},
    "12": {"batch_size": 1},
    "13": {"batch_size": 1},
    "14": {"batch_size": 1},
    "15": {"batch_size": 1},
    "16": {"batch_size": 1},
}

deep_supr_num = {
    "11": 3,
    "12": 3,
    "13": 3,
    "14": 3,
    "15": 3,
    "16": 3,
}