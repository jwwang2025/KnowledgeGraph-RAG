import json
from flask import Blueprint, jsonify

from app.kg import load_knowledge_graph


mod = Blueprint('graph', __name__, url_prefix='/graph')


@mod.route('/', methods=['GET'])
def graph():
    return jsonify({
        'data': load_knowledge_graph(),
        'message': 'You Got It!'
    })
