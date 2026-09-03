"""命名实体识别（PaddleNLP Taskflow）。"""
from paddlenlp import Taskflow


class Ner:
    def __init__(self):
        self.model = Taskflow("ner", task_path="weights/model_41_100")

    def predict(self, text):
        return self.model(text)

    def get_entities(self, text, etypes=None):
        """获取句子中指定类型的实体（只做一次预测，按类型过滤）"""
        if etypes is None:
            etypes = [None]

        results = self.predict(text)
        entities = []
        for etype in etypes:
            for ent, et in results:
                if not etype or etype in et:
                    entities.append(ent)
        return entities
