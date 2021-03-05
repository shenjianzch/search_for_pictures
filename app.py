from flask import Flask,request, send_file, jsonify,Response
from flask_restful import reqparse
import redis
import pymongo
import os
import json
from bson import json_util
from keras.applications.vgg16 import VGG16
from preprocessor.vggnet import vgg_extract_feat, VGGNet
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
from encoder.utils import feature_extract
from common.config import REDIS_NAME, REDIS_URI, REDIS_PORT, UPLOAD_PATH, DEFAULT_TABLE, THREAD_NUM, MONGODB_COLLECTION_NAME
from service.train import curd
from service.search import op_search
from service.delete import delete_collection
from service.threadpool import thread_do
from encoder.utils import filter_img
from service.count import count_tab
from service.partition import create_p
from service.common import collection_info, create_collection
import zipfile
# tensorflow 的设置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
global sess
sess = tf.Session(config=config)
set_session(sess)
input_shape = (224, 224, 3)


app = Flask(__name__)
model = None
r = None
graph = None
mongoCol = None


def load_mode():
    global graph
    graph = tf.get_default_graph()
    global model
    global r
    global mongoCol
    model = VGG16(weights='imagenet',
                  input_shape=input_shape,
                  pooling='max',
                  include_top=False)
    r = redis.Redis(host=REDIS_URI, port=REDIS_PORT, decode_responses=True)
    mongo_client = pymongo.MongoClient(host='localhost', port=27017)
    mongodb = mongo_client[REDIS_NAME]
    mongoCol = mongodb[MONGODB_COLLECTION_NAME]


@app.route('/')
def index():
    list = mongoCol.find()
    return Response(json_util.dumps(list),mimetype='application/json')


@app.route('/import_zip', methods=['post'])
def import_zip():
    file = request.files.get('file')
    if not file:
        return '参数file必填', 400
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_PATH, filename)
    file.save(path)
    print(filename.split('.')[0])
    zip_file = zipfile.ZipFile(path)
    zip_list = zip_file.namelist()
    for names in zip_list:
        if names[0] == '_':
            continue
        zip_file.extract(names, UPLOAD_PATH)
    zip_file.close()
    # 删除压缩包
    os.remove(path)
    # 开始处理压缩包里的图片
    img_dir_path = os.path.join(UPLOAD_PATH, filename.split('.')[0])
    thread_do(THREAD_NUM, img_dir_path, r)
    print('开始图片提取特征值了')
    return jsonify({'success': True})


# os.listdir 列出目录路径
@app.route('/upload', methods=['POST'])
def upload_img():
    file = request.files.get('file')
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_PATH, filename)
    file.save(file_path)
    # print(file_path, 'file_path')
    feat, img_name = feature_extract(file_path, VGGNet())
    curd(feat, img_name, r)
    # print(feat, img_name, 'lallalala')
    return 'ok', 200


@app.route('/search', methods=['post'])
def search_img():
    args = reqparse.RequestParser().\
        add_argument('table', type=str).\
        add_argument('num').\
        parse_args()
    table_name = args['table']
    top_k = args['num']
    if not table_name:
        table_name = DEFAULT_TABLE
    if not top_k:
        top_k = 10
    file = request.files.get('file')
    if not file:
        return '请上传文件', 400
    if not file.name:
        return '请保证文件有文件名', 400
    filename = secure_filename(file.name)
    file_path = os.path.join(UPLOAD_PATH, filename)
    file.save(file_path)
    res_id, res_distance = op_search(table_name, file_path, top_k, model, graph, sess)
    return jsonify(res_id), 200


@app.route('/collection/<table>', methods=['delete'])
def delete_index(table):
    if not table:
        return '参数错误', 400
    status = delete_collection(table)
    if not status:
        return '数据集合不存在', 400
    if status:
        return 'ok', 200


@app.route('/collection', methods=['post'])
def create_coll():
    args = reqparse.RequestParser().\
        add_argument('table', type=str).\
        parse_args()
    if not args['table']:
        return '参数错误', 400
    blen, status = create_collection(args['table'])
    if not blen:
        return '表已经存在', 400
    else:
        return '创建成功', 200


@app.route('/count', methods=['get'])
def get_count():
    args = reqparse.RequestParser().\
        add_argument('table', type=str).\
        parse_args()
    if not args['table']:
        return '参数错误', 400
    bool, num = count_tab(args['table'])
    if not bool:
        return '表不存在', 400
    else:
        return jsonify({'num': num}), 200


@app.route('/partition', methods=['post'])
def create_partition():
    args = reqparse.RequestParser().\
        add_argument('partition', type=str). \
        add_argument('table', type=str). \
        parse_args()
    if not args['partition'] or not args['table']:
        return '参数错误', 400
    status = create_p(args['table'], args['partition'])
    code = status.code
    message = status.message
    if code == 0:
        return '创建成功', 200
    else:
        return jsonify({'code': code, 'message': message}), 400


@app.route('/collection', methods=['get'])
def get_collection_info():
    args = reqparse.RequestParser().\
        add_argument('table', type=str). \
        parse_args()
    if not args['table']:
        return '参数错误', 400
    bool, status, info = collection_info(args['table'])
    if not bool:
        return '表不存在', 400
    print(status, 'ffffffff')
    if status.code == 0:
        return Response(json.dumps(info, default=lambda obj: obj.__dict__), mimetype='application/json'), 200
    else:
        return status.message, 400


if __name__ == '__main__':
    load_mode()
    app.run(host="0.0.0.0", port='6060')

