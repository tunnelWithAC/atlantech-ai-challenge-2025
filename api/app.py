from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample data - in a real application, this would come from a database
office_scores = {
    "google": [
        {"name": "Transit Accessibility", "score": 85},
        {"name": "Walking Distance", "score": 92},
        {"name": "Frequency of Service", "score": 78}
    ],
    "microsoft": [
        {"name": "Transit Accessibility", "score": 76},
        {"name": "Walking Distance", "score": 88},
        {"name": "Frequency of Service", "score": 81}
    ],
    "apple": [
        {"name": "Transit Accessibility", "score": 90},
        {"name": "Walking Distance", "score": 85},
        {"name": "Frequency of Service", "score": 95}
    ]
}

@app.route('/score', methods=['GET'])
def get_score():
    office_name = request.args.get('office_name', '').lower()
    
    if not office_name:
        return jsonify({"error": "office_name parameter is required"}), 400
    
    if office_name not in office_scores:
        return jsonify({"error": f"No data found for office: {office_name}"}), 404
    
    return jsonify(office_scores[office_name])

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Welcome to the Office Score API",
        "endpoints": {
            "GET /score": "Get scores for an office (required parameter: office_name)"
        },
        "available_offices": list(office_scores.keys())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 