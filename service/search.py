import logging
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index
from preprocessor.vggnet import vgg_extract_feat
import redis
from common.config import REDIS_NAME, REDIS_URI, REDIS_PORT, UPLOAD_PATH, DEFAULT_TABLE


def filter_data(vids):
    r = redis.Redis(host=REDIS_URI, port=REDIS_PORT, decode_responses=True)
    res = []
    for i in vids:
        val = r.hget(REDIS_NAME,i)
        print(val, i, 'redis缓存数据库中的数据')
        if val:
            res.append(val)
    return res


def op_search(table_name, img_path, top_k, model, graph, sess):
    try:
        feats = []
        clinet = milvus_client()
        feat = vgg_extract_feat(img_path, model, graph, sess)
        feats.append(feat)
        _, vectors = search_vectors(clinet, table_name, feats, top_k)
        print(vectors,'引擎查出来返回的值')
        vids = [x.id for x in vectors[0]]
        res_id = filter_data(vids)
        res_distance = [x.distance for x in vectors[0]]
        return res_id, res_distance
    except Exception as e:
        logging.error(e)
        return '发生错误'.format(e)
