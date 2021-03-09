import logging
from indexer.index import milvus_client, get_collection_info, has_table, create_table, get_table_list, get_list_partitions


def collection_info(table_name):
    client = milvus_client()
    tablestatus, ok = has_table(client, table_name)
    if not ok:
        return False, '', ''
    status, info = get_collection_info(client,table_name)
    return True, status, info


def create_collection(table_name):
    print(table_name,'1111')
    client = milvus_client()
    tablestatus, ok = has_table(client, table_name)
    print(tablestatus, ok)
    if ok:
        return False, ''
    status = create_table(client, table_name)
    return True, status


def get_collections():
    client = milvus_client()
    status, info = get_table_list(client)
    return status, info


def get_partitions_list(table_name):
    client = milvus_client()
    tablestatus, ok = has_table(client, table_name)
    if not ok:
        return ok, '', ''
    status, info = get_list_partitions(client, table_name)
    return ok, status, info
