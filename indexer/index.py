import logging as log
from milvus import Milvus, IndexType, MetricType, Status
from common.config import HOST, PORT, DIMENSION


def milvus_client():
    try:
        milvus = Milvus(host=HOST, port=PORT)
        return milvus
    except Exception as e:
        log.error(e)


def create_table(client, table_name=None, dimension=DIMENSION,
                 index_file_size=1024, metric_type=MetricType.L2):
    table_param = {
        'collection_name': table_name,
        'dimension': dimension,
        'index_file_size': index_file_size,
        'metric_type': metric_type
    }
    try:
        status = client.create_collection(table_param)
        return status
    except Exception as e:
        log.error(e)


def insert_vectors(client, table_name, vectors, partition_tag=None):
    if not client.has_collection(collection_name=table_name):
        log.error("集合 %s 不存在", table_name)
        return
    try:
        status, ids = client.insert(collection_name=table_name, records=vectors, partition_tag=partition_tag)
        return status, ids
    except Exception as e:
        log.error(e)


def create_index(client, table_name):
    param = {'nlist': 16384}
    # status = client.create_index(table_name, param)
    status = client.create_index(table_name, IndexType.IVF_FLAT, param)
    return status


def delete_table(client, table_name):
    status = client.drop_collection(collection_name=table_name)
    print(status)
    return status


# vectors->向量 查询可以通过指定某个分区进行查询
def search_vectors(client, table_name, vectors, top_k, partition_tags=None):
    search_param = {'nprobe': 16}
    status, res = client.search(collection_name=table_name, query_records=vectors, partition_tags=partition_tags, top_k=top_k, params=search_param)
    return status, res


def has_table(client, table_name):
    status = client.has_collection(collection_name=table_name)
    return status


def count_table(client, table_name):
    status, num = client.count_entities(collection_name=table_name)
    return num


#创建分区
def create_partition(client, table_name, partition_name):
    status = client.create_partition(table_name, partition_name)
    return status


#删除分区
def drop_partition(client, table_name, partition_name):
    status = client.drop_partition(collection_name=table_name, partition_tag=partition_name)
    return status


def get_collection_info(client, table_name):
    status, info = client.get_collection_info(table_name)
    return status, info


# 获取table 列表
def get_table_list(client):
    status, info = client.list_collections()
    return status, info


# 获取某个表的分区列表
def get_list_partitions(client, table_name):
    status, info = client.list_partitions(table_name)
    return status, info


# 查看是否有分区
def has_partition(client, table_name, partition):
    status, info = client.has_partition(table_name, partition)
    return status, info

