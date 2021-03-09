import logging
import random
from service.train import curd
from concurrent.futures import ThreadPoolExecutor
from encoder.utils import feature_extract
from preprocessor.vggnet import VGGNet
from encoder.utils import filter_img
from common.config import DEFAULT_TABLE


def thread_do(worker_num, filepath, mycol, model, partition=None, table_name=DEFAULT_TABLE, *args):
    if not table_name:
        table_name = DEFAULT_TABLE
    with ThreadPoolExecutor(max_workers=worker_num) as t:
        img_list = filter_img(filepath)
        for img in img_list:
            f = t.submit(thread_train, mycol, img, model, partition, table_name)
    # executor.submit(thread_train, *args)


def thread_train(mycol, file_path, model, partition=None, table_name=DEFAULT_TABLE, *args):
    print('开始提取了')
    if not table_name:
        table_name = DEFAULT_TABLE
    try:
        feat, img_name = feature_extract(file_path, model)
        bool, val = curd(feat, img_name, mycol, partition, table_name)
        return bool, val
    except Exception as e:
        logging.error(e)
        return '图片特征提取出错'.format(e)
