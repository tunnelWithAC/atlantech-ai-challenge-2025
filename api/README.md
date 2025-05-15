# Office Score API

A simple Flask API that returns scores for different offices based on the provided office name.

## Features

- Single endpoint `GET /score` that accepts an `office_name` parameter
- Returns an array of objects containing name and score for the specified office
- Containerized with Docker for easy deployment

## Setup Instructions

### Prerequisites

- Python 3.9+ installed
- pip (Python package manager)
- Docker (optional, for containerization)

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd office-score-api
   ```

2. Create and activate a virtual environment:
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. The API should be running at `http://localhost:5000`

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t office-score-api .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 office-score-api
   ```

3. The API should be accessible at `http://localhost:5000`

## API Documentation

### GET /score

Returns scores for the specified office.

**Query Parameters:**
- `office_name` (required): Name of the office to get scores for.

**Example Request:**
```
GET /score?office_name=google
```

**Example Response:**
```json
[
  {
    "name": "Transit Accessibility",
    "score": 85
  },
  {
    "name": "Walking Distance",
    "score": 92
  },
  {
    "name": "Frequency of Service",
    "score": 78
  }
]
```

**Error Responses:**

If no office_name is provided:
```json
{
  "error": "office_name parameter is required"
}
```

If the office is not found:
```json
{
  "error": "No data found for office: invalid_office"
}
```

## Available Offices

The sample data includes scores for the following offices:
- google
- microsoft
- apple

In a production environment, this would typically be stored in a database. 