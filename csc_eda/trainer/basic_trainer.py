#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/24 - 22:34

import dataclasses
import math
import os
import time
import traceback
from copy import deepcopy

import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, EvalPrediction, \
    TrainerCallback, is_torch_tpu_available, is_datasets_available

from transformers.trainer import Trainer
from typing import Optional, List, Union, Callable, Dict, Tuple, Any

from csc_eda.entity.csc_common_entity import ModelArguments, DataTrainingArguments
from csc_eda.utils.constants import Constants
from csc_eda.utils.file_utils import FileUtils
from csc_eda.utils.logger_utils import logger
from csc_eda.utils.time_utils import TimeUtils


class BaseTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            data_args: DataTrainingArguments = None,
            model_args: ModelArguments = None,
            run_log=None,
    ):
        super().__init__(model=model, args=args, data_collator=data_collator,
                         train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer,
                         model_init=model_init, compute_metrics=compute_metrics, callbacks=callbacks,
                         optimizers=optimizers)

        self.data_args = data_args
        self.model_args = model_args
        self.task_name = self.model_args.task_name
        self.show_info = self.model_args.show_info
        self.eval_ture_input = self.model_args.eval_ture_input
        self.model_name = self.model_args.model_name
        self.run_log = run_log
        self.output_dir = self.args.output_dir

        self.eval_file_path = self.data_args.eval_data_file
        self.add_default_post_process = self.model_args.add_default_post_process

        self.current_best_metric = 0 if self.args.greater_is_better else 1e9
        self.metric_show_file_name = f"{self.output_dir}/metric_show.txt"


        ####################################################################
        # 自己的训练函数
        #
        ####################################################################
        self.custom_train = False
        self.custom_eval = False

    def save_best_model(self, result, metric_key_prefix: str = "eval", ):
        """
        保存最好的模型

        :param result:
        :param metric_key_prefix:
        :return:
        """
        best_metric_name = f"{metric_key_prefix}_{self.args.metric_for_best_model}"
        best_metric = result[best_metric_name] if best_metric_name in result else None
        if best_metric is not None and self.state.epoch is not None:
            best_flag = best_metric > self.current_best_metric if self.args.greater_is_better else best_metric < self.current_best_metric
            if best_flag:
                model_state_dict = self.model.state_dict()
                save_dir = f"{self.output_dir}/best_model"
                save_best_model_path = f"{save_dir}/pytorch_model.bin"
                FileUtils.check_file_exists(save_best_model_path)
                torch.save(model_state_dict, save_best_model_path)

                # copy vocab and config file
                file_name_list = ["vocab.txt", "config.json", "special_tokens_map.json", "tokenizer_config.json"]
                for file_name in file_name_list:
                    file_path = f"{self.model_args.model_name_or_path}/{file_name}"
                    FileUtils.copy_file(file_path, save_dir)
                self.current_best_metric = best_metric
                logger.info(
                    f"保存最好的模型：{self.current_best_metric} - {self.state.best_metric} - {save_best_model_path} ")

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_metrics = None
        eval_begin_time = TimeUtils.now_str()

        self.save_log(eval_begin_time)

        if self.custom_eval:
            eval_metrics = self.evaluate_custom(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                                metric_key_prefix=metric_key_prefix)
        else:
            # 默认的评估过程
            eval_metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                                            metric_key_prefix=metric_key_prefix)

        # add save the eval metric to db
        eval_end_time = TimeUtils.now_str()

        all_metrics = {
            "eval_metrics": eval_metrics,
            "eval_begin_time": eval_begin_time,
            "eval_end_time": eval_end_time,
            "eval_use_time": TimeUtils.calc_diff_time(eval_begin_time, eval_end_time),
            "total": len(eval_dataset) if eval_dataset is not None else 0
        }

        try:
            self.save_eval_predict_to_db(all_metrics)

            ######################################################################
            # 指标汇总
            self.save_best_model(eval_metrics)
        except Exception as e:
            logger.warn(f"处理异常：{e}")
            # traceback.print_exc()

        return eval_metrics

    def evaluate_custom(self,
                        eval_dataset: Optional[Dataset] = None,
                        ignore_keys: Optional[List[str]] = None,
                        metric_key_prefix: str = "eval",
                        ) -> Dict[str, float]:
        pass

    def save_metric_to_file(self, all_metric, file_name):
        """
        保存评估指标到本地

        :param all_metric:
        :param file_name:
        :return:
        """
        # 添加其他的log
        run_log = []

        FileUtils.dump_json(file_name, all_metric, show_info=self.show_info)
        output_dir = os.path.join(self.output_dir, "eval_metric")
        add_suffix = self.get_current_epoch_and_step()
        eval_result_path = f"{output_dir}/eval_{add_suffix}_metric.json"
        FileUtils.dump_json(eval_result_path, all_metric)
        # logger.info(f"all_metric: {all_metric}")

        return run_log

    def get_result_metric_file_name(self, metric_key_prefix: str = "eval", ):
        """
        获取 预测的metric 的文件名称
        :param metric_key_prefix:
        :return:
        """

        run_result_metric_file = f"{Constants.DATA_DIR_JSON_RUN_METRIC}/{TimeUtils.get_time()}/{self.task_name}_{metric_key_prefix}_metric_{TimeUtils.now_str_short()}.json"
        return run_result_metric_file

    def get_copy_eval_file_to_separate_dir(self):
        """
        复制 评估过程文件到单独的文件夹

        :param all_metric:
        :return:
        """
        current_epoch = self.state.epoch
        epoch = current_epoch if current_epoch is not None else 0

        add_suffix = self.get_current_epoch_and_step()
        output_dir = f"{self.output_dir}/eval_file/{int(epoch)}/{add_suffix}"

        if self.state.epoch is None and self.state.global_step == 0:
            output_dir = self.output_dir

        return output_dir

    def get_current_epoch_and_step(self):
        current_epoch = self.state.epoch
        epoch = current_epoch if current_epoch is not None else 0

        output_dir = f"{epoch:.2f}_{self.state.global_step}"
        return output_dir

    def save_log(self, msg_list, mode='a'):
        """
        保存日志到本地日志文件

        :param msg_list:
        :param mode:
        :return:
        """
        if isinstance(msg_list, str):
            msg_list = [msg_list, "\n"]
        FileUtils.save_to_text(self.metric_show_file_name, "\n".join(msg_list), mode=mode)

    def is_model_training(self):
        return self.model.training

    def save_eval_predict_to_db(self, all_metrics, metric_key_prefix: str = "eval", ):
        pass

    def save_metric_to_db(self, all_metric, need_save_file=False,
                          run_log=None, run_type="成功", ):
        """
        保存评估指标到数据库
        :return:
        """
        db_dao = DbModelDao()

        if self.model_args.run_type != "成功":
            run_type = self.model_args.run_type

        # file_name = self.data_args.train_data_file
        file_name = self.data_args.eval_data_file
        # file_name = self.data_args.test_data_file
        # 查询文件
        run_model_file = db_dao.session.query(TModelFile).filter_by(file_path=file_name).first()

        # model_type = "CLS"
        model_type = "STS"

        if run_model_file is None:
            file_desc = "文本分类,CLS"
            # file_desc = f"文本匹配"
            file_type = "test"
            model_type = self.model_args.model_type
            model_name = self.model_args.model_name
            global_file_type = self.model_args.task_name
            dataset_name = self.data_args.dataset_name
            # save_file_name = f'{global_file_type}_{FileUtils.get_file_name(file_name)}_{file_type}'
            save_file_name = f'{dataset_name}_{file_type}'
            file_path = file_name
            run_model_file = TModelFile(model_type="STS", model_name=model_name, file_name=save_file_name,
                                        file_desc=f'{global_file_type},开源数据集,{file_desc}', file_type=file_type,
                                        file_path=file_path,
                                        total_line=FileUtils.get_file_line(file_path),
                                        file_size=FileUtils.get_file_size(file_path))
            if need_save_file:
                # 是否保存文件到数据库
                db_dao.insert(run_model_file)

        run_name = f"{run_model_file.file_name}_{FileUtils.get_file_name(all_metric['metric_file'])}"
        new_model_run = TModelRun(
            run_name=run_name,
            model_name_or_path=all_metric['model_name_or_path'],
            model_type=run_model_file.model_type, model_name=self.model_args.model_name,
            file_id=run_model_file.id, file_name=run_model_file.file_name,
            run_begin_time=all_metric["eval_begin_time"],
            run_end_time=all_metric["eval_end_time"],
            run_time=int(all_metric["use_time"]),
            run_type=run_type,
            run_log=",".join(run_log),
            run_log_file=logger.name,
            detect_f1=all_metric["run_metric_validate"]["metric"]["loss"],
            correct_f1=all_metric["run_metric_validate"]["metric"]["eval_info"]["f1"],
            precision_score=all_metric["run_metric_validate"]["metric"]["eval_info"]["precision"],
            recall=all_metric["run_metric_validate"]["metric"]["eval_info"]["recall"],
            run_metric=all_metric)
        db_dao.insert(new_model_run)

        # 拷贝执行文件到指定目录
        # DbModelService.copy_run_model_to_dir(run_name=f"%{run_name}%")

        # all_show_file = f"{Constants.DATA_DIR_NER_CHECK}/{TimeUtils.get_time()}/predict_metric_{TimeUtils.now_str_short()}.json"
        # all_show_metric["metric_file"] = all_show_file
        # FileUtils.dump_json(all_show_file, all_show_metric)
        # logger.info(f"保存运行 {self.model_name_or_path} 模型,执行文件：{self.file_name} 指标文件: {all_show_file}")
        # # logger.info(all_show_metric)
