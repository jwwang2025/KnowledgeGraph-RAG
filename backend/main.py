import os
import sys

# 添加项目根目录到路径，以便导入 config 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_dotenv, settings

load_dotenv()

settings.setup_cuda()

from app import apps
from app.model import start_model


if __name__ == '__main__':
    print("Starting model...")
    start_model()
    apps.secret_key = settings.SECRET_KEY.encode() if isinstance(settings.SECRET_KEY, str) else settings.SECRET_KEY
    apps.run(host=settings.SERVER_HOST, port=settings.SERVER_PORT, debug=settings.DEBUG, threaded=True)

