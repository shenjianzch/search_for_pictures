import os
from common.config import IMG_TYPE


def feature_extract(img_path, model):
    feats = []
    feat = model.vgg_extract_feat(img_path)
    image_name = os.path.split(img_path)[1]
    feats.append(feat)
    return feats, image_name


def filter_img(path):
    file_list = os.listdir(path)
    return [os.path.join(path, f) for f in file_list if f.split('.')[-1] in IMG_TYPE]
