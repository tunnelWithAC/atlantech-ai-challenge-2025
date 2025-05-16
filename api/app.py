from flask import Flask, request, jsonify
from ollama import chat
from ollama import ChatResponse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Building data with proximity scores to key locations
offices = {
    "bldg-1": {  # Downtown Office Tower
        "barna": 7,
        "knocknacarra": 9,
        "oranmore": 8
    },
    "bldg-2": {  # Riverside Apartments
        "barna": 8,
        "knocknacarra": 7,
        "oranmore": 6
    },
    "bldg-3": {  # Central Library
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    },
    "bldg-4": {  # City Hospital
        "barna": 7,
        "knocknacarra": 8,
        "oranmore": 8
    },
    "bldg-5": {  # University Campus Center
        "barna": 6,
        "knocknacarra": 8,
        "oranmore": 7
    },
    "bldg-6": {  # Tech Innovation Hub
        "barna": 5,
        "knocknacarra": 7,
        "oranmore": 8
    },
    "bldg-7": {  # Westside Shopping Mall
        "barna": 7,
        "knocknacarra": 8,
        "oranmore": 6
    }
}

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running", "available_endpoints": ["/prompt", "/score"]})

@app.route('/prompt', methods=['GET'])
def answer_prmopt():
    building_id = request.args.get('office_name', '')
    print(f"Received request for building ID: '{building_id}'")
    print(f"Available building IDs: {list(offices.keys())}")
    
    if not building_id:
        return jsonify({"error": "office_name parameter is required"}), 400
    
    if building_id not in offices:
        return jsonify({"error": f"No data found for building: {building_id}"}), 404

    building_scores = offices[building_id]
    
    # Create a prompt that describes the building based on its scores
    prompt = f"""
    Please explain what makes this location great based on these proximity scores:
    - Proximity to Barna: {building_scores['barna']}/10
    - Proximity to Knocknacarra: {building_scores['knocknacarra']}/10
    - Proximity to Oranmore: {building_scores['oranmore']}/10
    
    Please provide a natural, enthusiastic response highlighting the location's strengths based on these proximity scores.
    """
    
    response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])

    result = {
        'content': response.message.content,
        'office_name': building_id,
        'scores': building_scores
    }
    return jsonify(result)


@app.route('/score', methods=['GET'])
def get_score():
    building_id = request.args.get('office_name', '').lower()
    
    if not building_id:
        return jsonify({"error": "office_name parameter is required"}), 400
    
    if building_id not in offices:
        return jsonify({"error": f"No data found for building: {building_id}"}), 404
    
    results = {
        'scores': offices[building_id]
    }
    return jsonify(results)


if __name__ == '__main__':
    app.run(port=5001, debug=True) 