"""图像搜索：基于关键词匹配的静态图库。"""

# 静态关键词 -> 图片 URL 映射（模块级常量，实例间共享）
IMAGE_PAIRS = {
    "江南大学": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411102806.png",
    "军舰": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411102751.png",
    "消防手套": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411101854.png",
    "灭火剂": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411104044.png",
    "灭火": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411102650.png",
    "潜水装具": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411102343.png",
    "消防水枪": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411102528.png",
    "潜水": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411104255.png",
    "消防呼吸器": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411103758.png",
    "损管尺": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411104425.png",
    "喷射泵": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411104459.png",
    "空气泡沫喷漆": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411104707.png",
    "火灾": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411214012.png",
    "潜水员": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411214038.png",
    "测深仪": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411214124.png",
    "舰艇声呐": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411214203.png",
    "潜水呼吸装置": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411214221.png",
    "舰艇发动机": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411214326.png",
    "鲨鱼": "https://xerrors.oss-cn-shanghai.aliyuncs.com/imgs/20230411214352.png",
}


class ImageSearcher:
    """按关键词前缀匹配返回图片 URL"""

    def __init__(self):
        self.image_pair = IMAGE_PAIRS

    def search(self, query):
        for key, value in self.image_pair.items():
            if key in query:
                return value
        return None
