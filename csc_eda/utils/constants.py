#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2022/12/24 - 17:29


class Constants(object):
    """
    常量工具类

    """
    WORK_DIR = "../../"

    # WORK_DIR = "/home/mqq/shenglei/csc/csc_scope"

    DATA_DIR = f"{WORK_DIR}/data"
    LOG_DIR = f"{WORK_DIR}/logs"

    # 日志相关
    LOG_FILE = f"{LOG_DIR}/run.log"
    LOG_LEVEL = "debug"


    # DATA JSON
    DATA_DIR_JSON_RUN_METRIC = f"{DATA_DIR}/json/run_metric"
    DATA_DIR_JSON_ANALYSIS = f"{DATA_DIR}/json/analysis"