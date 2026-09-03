import json
import os

# 知识图谱数据缓存：文件较大且每次检索/请求都会用到，
# 只在文件变更时重新加载，避免每次调用都重复读盘和解析
_graph_cache = {"mtime": None, "data": None}


def load_knowledge_graph():
    """加载知识图谱数据（带 mtime 缓存）"""
    path = 'data/knowledge_graph/knowledge_graph.json'
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {'nodes': [], 'links': [], 'sents': []}

    if _graph_cache["mtime"] != mtime or _graph_cache["data"] is None:
        with open(path, 'r') as f:
            _graph_cache["data"] = json.load(f)
        _graph_cache["mtime"] = mtime
    return _graph_cache["data"]


def search_node_item(user_input, lite_graph=None):
    data = load_knowledge_graph()

    if lite_graph is None:
        lite_graph = {
            'nodes': [],
            'links': [],
            'sents': []
        }

    DEEP = 1

    def match_node(node_name, keyword):
        return node_name in keyword or keyword in node_name

    search_nodes = [user_input]
    for d in range(DEEP):
        for serch_node in search_nodes:
            for edge in data['links']:
                source = data['nodes'][int(edge['source'])]
                target = data['nodes'][int(edge['target'])]
                if match_node(source['name'], serch_node) or match_node(target['name'], serch_node):
                    sent = data['sents'][edge['sent']]
                    if sent not in lite_graph['sents']:
                        edge['sent'] = len(lite_graph['sents'])
                        lite_graph['sents'].append(sent)
                    else:
                        edge['sent'] = lite_graph['sents'].index(sent)

                    if source not in lite_graph['nodes']:
                        source['id'] = len(lite_graph['nodes'])
                        lite_graph['nodes'].append(source)
                    else:
                        source['id'] = lite_graph['nodes'].index(source)

                    if target not in lite_graph['nodes']:
                        target['id'] = len(lite_graph['nodes'])
                        lite_graph['nodes'].append(target)
                    else:
                        target['id'] = lite_graph['nodes'].index(target)

                    edge['source'] = source['id']
                    edge['target'] = target['id']
                    lite_graph['links'].append(edge)

        if len(lite_graph['nodes']) == 0:
            break

        search_nodes = [node['name'] for node in lite_graph['nodes']]

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
