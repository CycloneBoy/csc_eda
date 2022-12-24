#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/24 - 22:02
import os
import shutil
from copy import deepcopy

from csc_eda.utils.logger_utils import logger


class FileUtils(object):
    """
    文件工具类
    """

    @staticmethod
    def get_content(path, encoding='gbk'):
        """
        读取文本内容
        :param path:
        :param encoding:
        :return:
        """
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            content = ''
            for l in f:
                l = l.strip()
                content += l
            return content

    @staticmethod
    def check_file_exists(filename, delete=False):
        """检查文件是否存在"""
        if filename is None:
            return False
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print("文件夹不存在,创建目录:{}".format(dir_name))
        return os.path.exists(filename)

    @staticmethod
    def save_to_text(filename, content, mode='w'):
        """
        保存为文本
        :param filename:
        :param content:
        :return:
        """
        FileUtils.check_file_exists(filename)
        with open(filename, mode, encoding='utf-8') as f:
            f.writelines(content)

    @staticmethod
    def read_to_text(path, encoding='utf-8'):
        """读取txt 文件"""
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
            return content

    @staticmethod
    def read_to_text_list(path, encoding='utf-8'):
        """
        读取txt文件,默认utf8格式,
        :param path:
        :param encoding:
        :return:
        """
        list_line = []
        if not os.path.exists(path):
            return list_line
        with open(path, 'r', encoding=encoding) as f:
            list_line = f.readlines()
            list_line = [row.rstrip("\n") for row in list_line]
            return list_line

    @staticmethod
    def copy_file(src, dst, cp_metadata=False):
        """
        拷贝文件
        :param src:
        :return:
        """
        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)

        if src is None or len(src) == 0 or not os.path.exists(src):
            return dst_file_name
        FileUtils.check_file_exists(os.path.join(dst, "tmp.txt"))

        if cp_metadata:
            # Copy files, but preserve metadata (cp -p src dst)
            shutil.copy2(src, dst)
        else:
            #  Copy src to dst. (cp src dst)
            shutil.copy(src, dst)

        file_name = os.path.basename(src)
        dst_file_name = os.path.join(dst, file_name)
        return dst_file_name

    @staticmethod
    def read_csc_one_example(example, tokenizer, sep='\t', need_tokenize=False):
        """
            读取一行

        :param example:
        :param tokenizer:
        :param sep:
        :param need_tokenize:
        :return:
        """
        example = example.strip()
        line_split = example.split(sep)

        if len(line_split) == 3:
            line_split = line_split[1:]

        # need_tokenize = False
        if need_tokenize:
            line_input_items = tokenizer.tokenize(line_split[0])
        else:
            line_input_items = line_split[0].split()

        if len(line_split) >= 2:
            if need_tokenize:
                line_label_items = tokenizer.tokenize(line_split[1])
            else:
                line_label_items = line_split[1].split()
        else:
            logger.info(f"数据集长度不一致：{line_split}")
            line_label_items = deepcopy(line_input_items)

        return line_input_items, line_label_items

    @staticmethod
    def read_csc_one_example_and_tokenize(example, tokenizer, block_size, sep='\t', need_tokenize=False):
        """
         读取一行  并 tokenize
        :return:
        """
        line_input_items, line_label_items = FileUtils.read_csc_one_example(example=example,
                                                                            tokenizer=tokenizer,
                                                                            sep=sep,
                                                                            need_tokenize=need_tokenize)

        input_token = ["[CLS]"] + line_input_items[:block_size - 2] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_token)
        label_token = ["[CLS]"] + line_label_items[:block_size - 2] + ["[SEP]"]
        label_ids = tokenizer.convert_tokens_to_ids(label_token)

        return line_input_items, input_ids, line_label_items, label_ids
