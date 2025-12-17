import json
import os

# 全局缓存，避免每次请求都重新读取文件
_graph_data_cache = None
_graph_data_file_path = None
_graph_data_mtime = None

def _load_graph_data():
    """加载知识图谱数据，使用缓存机制"""
    global _graph_data_cache, _graph_data_file_path, _graph_data_mtime
    
    # 确定数据文件路径（相对于 server 目录）
    if _graph_data_file_path is None:
        # 尝试多个可能的路径
        possible_paths = [
            'data/data.json',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data.json'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data.json'),
        ]
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                _graph_data_file_path = abs_path
                break
        
        if _graph_data_file_path is None:
            raise FileNotFoundError("无法找到 data/data.json 文件，请检查文件路径")
    
    # 检查文件是否被修改
    current_mtime = os.path.getmtime(_graph_data_file_path)
    if _graph_data_cache is None or _graph_data_mtime != current_mtime:
        print(f"[graph_utils] 加载知识图谱数据: {_graph_data_file_path}")
        with open(_graph_data_file_path, 'r', encoding='utf-8') as f:
            _graph_data_cache = json.load(f)
        _graph_data_mtime = current_mtime
        print(f"[graph_utils] 知识图谱数据加载完成: {len(_graph_data_cache.get('nodes', []))} 个节点, {len(_graph_data_cache.get('links', []))} 条边")
    
    return _graph_data_cache


def search_node_item(user_input, lite_graph=None):
    print(f"[graph_utils] 搜索节点: {user_input}")
    data = _load_graph_data()

    if lite_graph is None:
        lite_graph = {
            'nodes': [],
            'links': [],
            'sents': []
        }

    # 利用thefuzz库来选取最相近的节点
    # node_names = [node['name'] for node in data['nodes']]
    # user_input = process.extractOne(user_input, node_names)[0]

    DEEP = 1

    # search node
    search_nodes = [user_input]
    for d in range(DEEP):
        for serch_node in search_nodes:
            for edge in data['links']:
                source = data['nodes'][int(edge['source'])]
                target = data['nodes'][int(edge['target'])]
                if source['name'] in serch_node or serch_node in source['name'] or target['name'] in serch_node or serch_node in target['name']:
                # if source['name'] == serch_node or target['name'] == serch_node:
                    sent = data['sents'][edge['sent']]
                    if sent not in lite_graph['sents']:
                        edge['sent'] = len(lite_graph['sents'])
                        lite_graph['sents'].append(sent)
                    else:
                        edge['sent'] = lite_graph['sents'].index(sent)

                    # 创建节点的副本，避免修改原始数据
                    source_copy = source.copy()
                    target_copy = target.copy()
                    
                    # 使用节点名称来查找，而不是对象比较
                    source_exists = False
                    target_exists = False
                    source_id = None
                    target_id = None
                    
                    for idx, node in enumerate(lite_graph['nodes']):
                        if node.get('name') == source['name']:
                            source_exists = True
                            source_id = idx
                        if node.get('name') == target['name']:
                            target_exists = True
                            target_id = idx
                    
                    if not source_exists:
                        source_copy['id'] = len(lite_graph['nodes'])
                        lite_graph['nodes'].append(source_copy)
                        source_id = source_copy['id']
                    else:
                        source_copy['id'] = source_id

                    if not target_exists:
                        target_copy['id'] = len(lite_graph['nodes'])
                        lite_graph['nodes'].append(target_copy)
                        target_id = target_copy['id']
                    else:
                        target_copy['id'] = target_id

                    edge_copy = edge.copy()
                    edge_copy['source'] = source_copy['id']
                    edge_copy['target'] = target_copy['id']
                    lite_graph['links'].append(edge_copy)

        if len(lite_graph['nodes']) == 0:
            break

        search_nodes = [node['name'] for node in lite_graph['nodes']]

    print(f"[graph_utils] 搜索完成: 找到 {len(lite_graph['nodes'])} 个节点, {len(lite_graph['links'])} 条边")
    return lite_graph


def convert_graph_to_triples(graph, entity=None):
    triples = []
    for link in graph['links']:
        source = graph['nodes'][link['source']]
        target = graph['nodes'][link['target']]

        if entity is not None:
            if entity in source['name'] or entity in target['name']:
                triples.append((source['name'], link["name"], target['name']))
        else:
            triples.append((source['name'], link["name"], target['name']))

    return triples