import os
import logging
from indexer.index import milvus_client, delete_table, has_table, delete_entity


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


def delete_imgs(table_name, ids, mycol):
    print(table_name,ids,'内部')
    if not table_name:
        return False
    try:
        client = milvus_client()
        status, ok = has_table(client, table_name)
        if not ok:
            return False, '表不存在'
        if ok:
            list=[]
            for id in ids:
                item = mycol.find_one({'img':id})
                if item:
                    list.append(int(item['id']))
            if list:
                dstatus = delete_entity(client, table_name, list)
                for id in ids:
                    mycol.delete_one({"img": id})
                if dstatus.code == 0:
                    return True, '删除成功'
            else:
                return True, '该图片未进行索引或已被删除'
    except Exception as e:
        logging.error(e)
        return False


