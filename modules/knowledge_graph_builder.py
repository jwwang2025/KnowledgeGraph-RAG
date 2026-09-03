import os
import json
import time
from modules.prepare.preprocess import process_text
from modules.prepare.manualkg import refine_knowledge_graph
from modules.prepare.process import uie_execute
from modules.prepare.filter import auto_filter

from modules.model_trainer import ModelTrainer

from modules.prepare import cprint as ct
from config.settings import settings

class KnowledgeGraphBuilder:

    def __init__(self, args) -> None:
        """

        文件的存储路径，以及一些参数的初始化

        """
        # self.args = args # 不能被序列化
        data_dir = settings.DATA_DIR
        self.data_dir = os.path.join(str(data_dir), args.project)
        self.text_path = str(settings.RAW_DATA_PATH)
        self.base_kg_path = os.path.join(self.data_dir, "base.json")
        self.refined_kg_path = os.path.join(self.data_dir, "base_refined.json")
        self.filtered_kg_path = os.path.join(self.data_dir, "base_filtered.json")

        # 优先使用本地模型路径，如果不存在则使用配置中的模型名称（会从HuggingFace下载）
        bert_model_name = settings.BERT_MODEL_NAME
        local_model_path = os.path.join("models", bert_model_name.split("/")[-1])
        if os.path.exists(local_model_path) and os.path.exists(os.path.join(local_model_path, "tokenizer_config.json")):
            self.model_name_or_path = local_model_path
            print(f"使用本地模型路径: {self.model_name_or_path}")
        else:
            self.model_name_or_path = bert_model_name  # 使用配置中的模型名称
            print(f"使用模型名称（将从HuggingFace下载）: {self.model_name_or_path}")
        self.version = 0
        self.kg_paths = []
        self.gpu = args.gpu if args.gpu else settings.DEFAULT_GPU

        os.makedirs(self.data_dir, exist_ok=True)


    def run_iteration(self):
        """
        运行一次迭代，包括：
        1. 读取上一次迭代的结果，如果是第一次迭代，则读取 base_kg_path
        2、训练，对齐和扩展
        3、保存结果
        """

        print(ct.green("Start Running Iteration:"), ct.yellow(f"v{self.version}"))

        cur_data_path = self.kg_paths[-1] if self.version > 0 else self.refined_kg_path
        cur_out_path = os.path.join(self.data_dir, f"iteration_v{self.version}")

        print(ct.green("Current Data Path:"), ct.yellow(cur_data_path), ct.red(cur_out_path))

        trainer = ModelTrainer(cur_data_path, cur_out_path, self.model_name_or_path, self.gpu)

        # 判断是否已经训练过了，毕竟这个地方可能会出问题的
        if not os.path.exists(trainer.prediction):
            trainer.train_and_test()
            assert os.path.exists(trainer.prediction), ct.red("Prediction file not found! It seems that the training process failed.")
            self.save()
        else:
            print(ct.yellow("Prediction file already exists, skip training."))

        trainer.relation_align()
        trainer.refine_and_extend()

        self.kg_paths.append(trainer.final_knowledge_graph)
        self.save()
        self.version += 1

    def extend_ratio(self):
        """用于计算扩展的比例，如果扩展的比例小于 10%，则认为已经收敛"""

        if self.version < 2 or len(self.kg_paths) < 2:
            return 1.0

        pre_kg = self.kg_paths[-2]
        cur_kg = self.kg_paths[-1]

        total_rel = 0
        extend_rel = 0
        with open(pre_kg, 'r', encoding='utf-8') as f_pre, open(cur_kg, 'r', encoding='utf-8') as f_cur:
            pre_lines = [json.loads(line) for line in f_pre.readlines()]
            cur_lines = [json.loads(line) for line in f_cur.readlines()]

            assert len(pre_lines) == len(cur_lines)

            for pre_line, cur_line in zip(pre_lines, cur_lines):
                pre_rels = pre_line['relationMentions']
                cur_rels = cur_line['relationMentions']

                total_rel += len(pre_rels)
                extend_rel += len(cur_rels) - len(pre_rels)
                assert len(pre_rels) <= len(cur_rels)

        return extend_rel / total_rel


    def get_base_kg_from_txt(self):
        """通过 UIE 获取基础知识图谱，并将其格式化为 SPN 风格 
        input: self.text_path
        output: self.refined_kg_path
        """
        texts = process_text(self.text_path, 480)

        # 如果 base_kg_path 已存在则跳过 UIE，删除该文件可重新抽取
        if not os.path.exists(self.base_kg_path):
            all_items = uie_execute(texts)
            with open(self.base_kg_path, 'w', encoding='utf-8') as f:
                for item in all_items:
                    f.writelines(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            print(f"Base KG already exists in {self.base_kg_path}, skip UIE.")

        with open(self.base_kg_path, 'r', encoding='utf-8') as f:
            all_items = [json.loads(line) for line in f.readlines()]
            filtered_items = auto_filter(all_items, self.model_name_or_path)

        with open(self.filtered_kg_path, 'w', encoding='utf-8') as f:
            for item in filtered_items:
                f.writelines(json.dumps(item, ensure_ascii=False) + "\n")


        # 人工筛选需要加断点，所以需要一边做一边保存
        refine_knowledge_graph(self.filtered_kg_path, self.refined_kg_path, fast_mode=True)

    def save(self, save_path=None):
        if save_path is None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            histaory_dir = os.path.join(self.data_dir, "history")
            os.makedirs(histaory_dir, exist_ok=True)
            save_path = os.path.join(histaory_dir, f"{timestr}_iter_v{self.version}.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=4)

        print(f"Save state to {save_path}.")
        print(ct.blue(f"Current version: {self.version}"))
        print("You can use", ct.green(f"--resume {save_path}"), "to continue training.\n")

    def load(self, load_path=None):
        with open(load_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.__dict__.update(state)
