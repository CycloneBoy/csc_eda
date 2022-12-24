#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/24 - 22:05
from unittest import TestCase

from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoModelForTokenClassification

from csc_eda.our_dataset.csc_dataset_bert import CscBertDataset
from csc_eda.utils.file_utils import FileUtils


class ModelDatasetTest(TestCase):

    def setUp(self):
        self.batch_size = 2
        self.max_length = 130
        self.use_fp16 = True

    def test_base_model(self):
        model_name_or_path = "/home/mqq/shenglei/data/nlp/bert/ernie/ernie_3.0_base_ch"
        eval_data_file = "/home/mqq/shenglei/csc/PLOME_finetune_pytorch/datas/test_1100.txt"

        num_labels = len(FileUtils.read_to_text_list(f"{model_name_or_path}/vocab.txt"))

        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        dataset_type = "dev"
        dataset = CscBertDataset(file_path=eval_data_file, tokenizer=tokenizer,
                                 block_size=self.max_length, dataset_type=dataset_type)
        print(dataset[0])

        collate_fn = dataset.get_collate_fn()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(dataloader))
        print(f"total:{len(dataloader)}")
        print(batch)

        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, num_labels=num_labels, )

        output = model(**batch)
        print(output)


def demo_dataset():
    model_runner = ModelDatasetTest()


if __name__ == '__main__':
    demo_dataset()
