import logging
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index, has_partition
from preprocessor.vggnet import vgg_extract_feat
import redis
from common.config import REDIS_NAME, REDIS_URI, REDIS_PORT, UPLOAD_PATH, DEFAULT_TABLE
from bson import json_util


def filter_data(vids, mycol, vectors):
    res = []
    list = mycol.find({"id": {"$in": vids}}, {"_id": 0})
    for i in list:
        for d in vectors:
            # print(i,d,'iuio')
            if d.id == i['id']:
                res.append({'img': i['img'], 'distance': d.distance})
    return res


def op_search(table_name, img_path, top_k, model, graph, sess, mycol, partition=None):
    print(top_k,'top_k')
    try:
        feats = []
        client = milvus_client()
        if partition:
            status, ok = has_partition(client, table_name, partition)
            if not ok:
                return False, ''
        feat = vgg_extract_feat(img_path, model, graph, sess)
        feats.append(feat)
        _, vectors = search_vectors(client, table_name, feats, top_k, partition)
        vids = [x.id for x in vectors[0]]
        print(vectors, 'vectors[0]')
        # res_id = filter_data(vids, mycol)
        # res_distance = [x.distance for x in vectors[0]]
        res = filter_data(vids, mycol, vectors[0])
        return True, res
    except Exception as e:
        logging.error(e)
        return '发生错误'.format(e)
