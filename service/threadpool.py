import logging
import random
from service.train import curd
from concurrent.futures import ThreadPoolExecutor
from encoder.utils import feature_extract
from preprocessor.vggnet import VGGNet
from encoder.utils import filter_img
from common.config import DEFAULT_TABLE


def thread_do(worker_num, filepath, r, *args):
    print(r)
    with ThreadPoolExecutor(max_workers=worker_num) as t:
        img_list = filter_img(filepath)
        for img in img_list:
            f = t.submit(thread_train, r, img)
    # executor.submit(thread_train, *args)


def thread_train(r, file_path, table_name=DEFAULT_TABLE, *args):
    print('开始提取了')
    try:
        feat, img_name = feature_extract(file_path, VGGNet())
        curd(feat, img_name, r, table_name)
    except Exception as e:
        logging.error(e)
        return '图片特征提取出错'.format(e)
