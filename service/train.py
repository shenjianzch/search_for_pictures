import logging
import time
import zipfile
from common.config import DEFAULT_TABLE, REDIS_NAME
from indexer.index import milvus_client, create_table, insert_vectors, delete_table, search_vectors, create_index, has_table, has_partition, create_partition


def curd(vectors, img_name, mycol, partition=None, table_name=DEFAULT_TABLE):
    try:
        client = milvus_client()
        if not table_name:
            table_name = DEFAULT_TABLE
        status, ok = has_table(client, table_name)
        if not ok:
            print('开始创建table')
            create_table(client, table_name)
        if partition:
            status, ok = has_partition(client, table_name, partition)
            if not ok:
                create_partition(client, table_name, partition)
        status, id = insert_vectors(client, table_name, vectors, partition)
        # 存入缓存 以便后续进行反查
        # redis.hset(REDIS_NAME, id[0], img_name)
        mycol.insert_one({'id': id[0], 'img': img_name, 'table': table_name})
        create_index(client, table_name)
        print('OK 了')
        return True, ''
    except Exception as e:
        logging.error(e)
        return False, e

