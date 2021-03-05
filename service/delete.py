import os
import logging
from indexer.index import milvus_client, delete_table, has_table


def delete_collection(table_name):
    if not table_name:
        return False
    try:
        client = milvus_client()
        status, ok = has_table(client,table_name)
        if not ok:
            return False
        if ok:
            status = delete_table(client, table_name)
            if status.code == 0:
                return True
    except Exception as e:
        logging.error(e)
        return '错误'.format(e)

