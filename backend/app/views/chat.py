from flask import Response, request, Blueprint

from app.model import stream_predict

mod = Blueprint('chat', __name__, url_prefix='/chat')


@mod.route('/', methods=['GET'])
def chat_get():
    return "Chat Get!"


@mod.route('/', methods=['POST'])
def chat():
    request_data = request.get_json(silent=True) or {}
    prompt = request_data.get('prompt', '')
    history = request_data.get('history', [])

    return Response(response=stream_predict(prompt, history=history),
                    content_type='application/json', status=200)
