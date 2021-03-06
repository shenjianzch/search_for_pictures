import os

HOST = os.getenv('HOST', '192.168.71.149')
PORT = os.getenv('PORT', 19530)
DIMENSION = os.getenv('DIMENSION', 512)
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
UPLOAD_PATH = "/data/pic-server/search_for_pictures/tmp"
REDIS_NAME = "HCLC_IMG_COLLECTION"
REDIS_URI = os.getenv('REDIS_URI', '127.0.0.1')
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
IMG_TYPE = ["jpg", "png", "jpeg"]
THREAD_NUM = os.getenv("THREAD_NUM", 20)
MONGODB_COLLECTION_NAME = 'HCLC'
MONGODB_URI = os.getenv('MONGODB_URI', '192.168.70.198')
MONGODB_PORT = os.getenv('MONGODB_PORT', 27017)

