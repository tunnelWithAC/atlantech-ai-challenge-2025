from flask import Flask, request, jsonify
from ollama import chat
from ollama import ChatResponse

app = Flask(__name__)


offices = {
    "platform94": {
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    },
    "portershed": {
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    },
    "parkmore": {
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    }
}


@app.route('/prompt', methods=['GET'])
def answer_prmopt():
    prompt = request.args.get('prompt', '').lower() 
    if not prompt:
        return jsonify({"error": "prompt parameter is required"}), 400
    
    response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
    ])

    result = {
        'content': response.message.content
    }
    return jsonify(result)


@app.route('/score', methods=['GET'])
def get_score():
    office_name = request.args.get('office_name', '').lower()
    
    if not office_name:
        return jsonify({"error": "office_name parameter is required"}), 400
    
    if office_name not in offices.keys():
        return jsonify({"error": f"No data found for office: {office_name}"}), 404
    
    results = {
        'scores': offices[office_name]
    }
    return jsonify(results)


if __name__ == '__main__':
    app.run(port=8080, debug=True) 