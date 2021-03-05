import logging
from indexer.index import milvus_client, create_table, create_partition, has_table


def create_p(table_name, partition_name):
    client = milvus_client()
    try:
        status, ok = has_table(client, table_name)
        if not ok:
            create_table(client, table_name)

        statusp = create_partition(client, table_name, partition_name)
        return statusp
    except Exception as e:
        logging.error(e)
        return '发生错误'.format(e)





