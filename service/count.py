import logging
from indexer.index import milvus_client, delete_table, has_table, count_table


def count_tab(table_name):
    if not table_name:
        return False
    try:
        client = milvus_client()
        status, ok = has_table(client, table_name)
        if not ok:
            return False, ''
        val = count_table(client, table_name)
        return True, val
    except Exception as e:
        logging.error(e)
        return '发生错误'.format(e)
