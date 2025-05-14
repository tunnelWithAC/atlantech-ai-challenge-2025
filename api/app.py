from flask import Flask, request, jsonify

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

@app.route('/score', methods=['GET'])
def get_score():
    office_name = request.args.get('office_name', '').lower()
    
    if not office_name:
        return jsonify({"error": "office_name parameter is required"}), 400
    
    if office_name not in offices.keys():
        return jsonify({"error": f"No data found for office: {office_name}"}), 404
    
    return jsonify(offices[office_name])

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