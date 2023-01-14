from pathlib import Path
from typing import Set

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split


_URL = ["https://ytlin.s3.ap-northeast-1.amazonaws.com/data/huggingface_datasets/ROCStories/ROCStories2016.csv",
        "https://ytlin.s3.ap-northeast-1.amazonaws.com/data/huggingface_datasets/ROCStories/ROCStories2017.csv"]


class RocStory(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default")]

    turn_sep = "<|endoftext|>"

    def _info(self):
        features = datasets.Features(
            {
                # origianl features
                "storyid": datasets.Value("string"),
                "storytitle": datasets.Value("string"),
                "sentence1": datasets.Value("string"),
                "sentence2": datasets.Value("string"),
                "sentence3": datasets.Value("string"),
                "sentence4": datasets.Value("string"),
                "sentence5": datasets.Value("string"),
                # model-specific
                "storytitle+endoftext": datasets.Value("string"),
                "story": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        # data_paths = self.dl_manager.download_and_extract(_URL)
        data_paths = [
            "/home/lidong1/CodeRepo/data/roc_story/ROCStories2016.csv",
            "/home/lidong1/CodeRepo/data/roc_story/ROCStories2017.csv",
        ]
        df = pd.concat([pd.read_csv(data_path) for data_path in data_paths])
        storyids = df["storyid"].to_list()

        train_storyids, test_storyids = train_test_split(
            storyids, test_size=0.1, random_state=42
        )
        train_storyids = set(train_storyids)
        test_storyids = set(test_storyids)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"df": df, "story_ids": train_storyids},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"df": df, "story_ids": test_storyids},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"df": df, "story_ids": test_storyids},
            ),
        ]

    def _generate_examples(self, df: pd.DataFrame, story_ids: Set):
        id_ = -1
        for row in df.to_dict(orient="records"):
            if row["storyid"] not in story_ids:
                continue
            sentences = [
                row["sentence1"],
                row["sentence2"],
                row["sentence3"],
                row["sentence4"],
                row["sentence5"],
            ]
            row["storytitle+endoftext"] = row["storytitle"] + f" {self.turn_sep}"
            row["story"] = f" {self.turn_sep} ".join(sentences)
            id_ += 1
            yield id_, row
