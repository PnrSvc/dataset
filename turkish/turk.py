# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""turkk."""


import csv

import datasets
from datasets.tasks import TextClassification


_CITATION = """\
@inproceedings{Casanueva2020,
    author      = pnr,
    title       = {sentiment},
    year        = {2022},
    month       = {mar},
    note        = {Data available at https://github.com/PnrSvc/dataset},
    url         = {a},
    booktitle   = {a}
}
""" 

_DESCRIPTION = """\
description
"""

_HOMEPAGE = "https://github.com/PnrSvc/dataset"


_TRAIN_DOWNLOAD_URL = (
    "https://github.com/PnrSvc/dataset/blob/main/turkish/train.csv"
)
_TEST_DOWNLOAD_URL = "https://github.com/PnrSvc/dataset/blob/main/turkish/test.csv"


class Datas(datasets.GeneratorBasedBuilder):
    """datas dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "label": datasets.Value("string"),
                "target": datasets.features.ClassLabel(
                    names=[
                         "negative",
                         "neutral",
                         "positive"
                    ]
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[TextClassification(text_column="label", label_column="target")],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding="utf-8") as f:
            csv_reader = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            # call next to skip header
            next(csv_reader)
            for id_, row in enumerate(csv_reader):
                label, target = row
                yield id_, {"text": label, "label": target}