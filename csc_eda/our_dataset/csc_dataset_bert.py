#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  :
# @Author: sl
# @Date  : 2022/12/24 - 16:48

import os
from copy import deepcopy
from typing import List, Dict, Union

import numpy as np
import torch
from attr import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from csc_eda.utils.file_utils import FileUtils
from csc_eda.utils.logger_utils import logger

"""

DATASET

bert

"""


class CscBertDataset(Dataset):
    """
    bert 数据集
    """

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, block_size: int = 128,
                 need_cache=False, input_examples=None, dataset_name="cls", show_number=3, eval_ture_input=False,
                 extract_detect_label=True,
                 read_all=False, need_tokenize=False,
                 label_list=None, sep="\t", *args, **kwargs):
        # print(file_path, os.path.isfile(file_path))
        assert os.path.isfile(file_path)
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.need_cache = need_cache
        self.dataset_name = dataset_name
        self.extract_detect_label = extract_detect_label

        self.input_examples: List = [] if input_examples is None else input_examples
        self.show_number = show_number
        self.eval_ture_input = eval_ture_input
        self.sep = sep
        self.need_tokenize = need_tokenize

        self.count = 0

        self.input_ids = []
        self.label_ids = []
        self.input_lens = []
        self.label_ids_detect = []
        self.pinyin_ids_ecspells = []

        self.input_file = None
        self.lines = None

        self.type_weight = {}
        self.read_all = read_all
        self.read_data_from_file()

    def read_data_from_file(self):
        if len(self.input_examples) == 0:
            self.input_file = open(self.file_path, encoding="utf-8")
            logger.info(f"开始加载数据: {self.file_path} ")
            self.lines = self.input_file.readlines()

            self.count = len(self.lines)
            logger.info(f"加载数据完毕: 总共：{self.count} 条 - {self.file_path}")
        else:
            self.lines = self.input_examples
            self.count = len(self.input_examples)

        if not self.read_all:
            logger.info(f"增量读取数据集")

            return

        for index, line in enumerate(tqdm(range(0, self.count))):
            line = self.lines[index]
            line = line.strip()
            line_split = line.split(self.sep)

            line_input_items = line_split[0].split()
            line_label_items = line_split[1].split()

            input_token = ["[CLS]"] + line_input_items[:self.block_size - 2] + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_token)
            label_token = ["[CLS]"] + line_label_items[:self.block_size - 2] + ["[SEP]"]
            label_ids = self.tokenizer.convert_tokens_to_ids(label_token)

            seq_len = len(input_ids) - 2

            self.input_ids.append(torch.LongTensor(input_ids))
            self.label_ids.append(torch.LongTensor(label_ids))
            # self.label_ids.append(torch.LongTensor(detect_label_ids))
            self.input_lens.append(seq_len)

    def extract_detect_label_ids(self, input_ids, label_ids):
        """
        提取检测的label

        :param input_ids:
        :param label_ids:
        :return:
        """
        # modify
        detect_label_ids = []
        detect_label_ids.append(0)
        for idx, src_token in enumerate(input_ids[1:-1]):
            tag_token = label_ids[idx + 1]
            if src_token == tag_token:
                detect_label_ids.append(0)
            else:
                detect_label_ids.append(1)
        detect_label_ids.append(0)
        return detect_label_ids

    def get_collate_fn(self, task_name="csc", ):
        collate_fn = DataCollatorCscBert(task_name=task_name)
        return collate_fn

    def get_one_input_ids(self, example):
        """
        增量获取
        :param example:
        :return:
        """
        line_input_items, input_ids, line_label_items, label_ids = FileUtils.read_csc_one_example_and_tokenize(
            example=example,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            sep=self.sep,
            need_tokenize=self.need_tokenize)

        seq_len = len(input_ids) - 2

        if len(input_ids) != len(label_ids):
            label_ids = deepcopy(input_ids)

        detect_label_ids = self.extract_detect_label_ids(input_ids, label_ids)

        data = {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(label_ids),
            "labels_detect": torch.LongTensor(detect_label_ids),
            "input_lens": seq_len
        }

        return data

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if self.read_all:
            data = {
                "input_ids": self.input_ids[index],
                "labels": self.label_ids[index],
                "input_lens": self.input_lens[index],
            }
        else:
            example = self.lines[index]
            data = self.get_one_input_ids(example)

        return data


@dataclass
class DataCollatorCscBert:
    task_name: str = "csc"

    # add_other_params = True
    add_other_params = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [feature["input_ids"] for feature in features]
        batch_labels = [feature["labels"] for feature in features]
        # input_lens = [feature["input_lens"] for feature in features]
        batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

        if "labels_detect" in features[0]:
            batch_labels_detect = [feature["labels_detect"] for feature in features]
            batch_labels_detect = pad_sequence(batch_labels_detect, batch_first=True, padding_value=0)

        # padding
        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
        batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0)
        batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
        # batch_attention_mask_label = pad_sequence(batch_attention_mask_label, batch_first=True, padding_value=0)

        assert batch_input_ids.shape == batch_labels.shape
        assert batch_input_ids.shape == batch_attention_mask.shape

        batch = {
            "input_ids": batch_input_ids,
            "labels": torch.LongTensor(batch_labels),
            "attention_mask": batch_attention_mask,
        }

        return batch
