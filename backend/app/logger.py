import logging
import logging.config
import os

FORMAT = '%(asctime)s %(levelname)s %(name)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('server')


def setup_logger(name: str = 'server', level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """配置并返回指定名称的 logger（可选输出到文件）"""
    log = logging.getLogger(name)
    log.setLevel(level)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter(FORMAT))
        log.addHandler(handler)
    return log


def get_logger(name: str = 'server') -> logging.Logger:
    """获取指定名称的 logger"""
    return logging.getLogger(name)
