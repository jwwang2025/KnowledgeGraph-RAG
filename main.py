import argparse
import os

# 加载配置系统
from config.settings import load_dotenv, settings

# 加载 .env 文件中的配置
load_dotenv()

# 设置 CUDA 环境变量
settings.setup_cuda()

from modules.knowledge_graph_builder import KnowledgeGraphBuilder


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="project_v1")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    parser.add_argument("--build-vector-index", action="store_true", help="build vector index only")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parser()
    
    # 单独构建向量索引模式
    if args.build_vector_index:
        from modules.vector_indexer import VectorIndexer
        indexer = VectorIndexer(project_name=args.project)
        indexer.index_all()
        print("向量索引构建完成！")
        exit(0)
    
    kg_builder = KnowledgeGraphBuilder(args)

    if args.resume is not None:
        kg_builder.load(args.resume)
        kg_builder.gpu = args.gpu # 这个是要换掉的

    else:
        # startup
        kg_builder.get_base_kg_from_txt()

    # iteration
    max_iteration = settings.MAX_ITERATION

    while kg_builder.version < max_iteration:
        kg_builder.run_iteration() # 迭代过程中会自动保存
        extend_ratio = kg_builder.extend_ratio()
        print(f"Extend Ratio: {extend_ratio}")

        if extend_ratio < settings.EXTEND_RATIO_THRESHOLD:
            print(f"Extend Ratio ({extend_ratio:.4f}) is below threshold ({settings.EXTEND_RATIO_THRESHOLD}), stop iteration.")
            break

    print("done!")