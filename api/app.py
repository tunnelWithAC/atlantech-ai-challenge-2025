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
    office_name = request.args.get('office_name', '').lower()
    if not office_name:
        return jsonify({"error": "office_name parameter is required"}), 400
    
    if office_name not in offices:
        return jsonify({"error": f"No data found for office: {office_name}"}), 404

    office_scores = offices[office_name]
    
    # Create a prompt that describes the office based on its scores
    prompt = f"""
    Please explain what makes the {office_name} office location great based on these scores:
    - Proximity to Barna: {office_scores['barna']}/10
    - Proximity to Knocknacarra: {office_scores['knocknacarra']}/10
    - Proximity to Oranmore: {office_scores['oranmore']}/10
    
    Please provide a natural, enthusiastic response highlighting the office's strengths based on these proximity scores.
    """
    
    response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    result = {
        'content': response.message.content,
        'office_name': office_name,
        'scores': office_scores
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