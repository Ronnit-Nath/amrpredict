import requests

# URL of the Flask app
url = "http://localhost:5000/predict"

# Example data to be sent to the predict endpoint
data = {
    "sequences": ["AGCTAGCTAG", "TCGATCGATC", "ATCGATCGAT", "GATCGATCGA", "CGATCGATCG"],
    "model_type": "logistic"  # or "random_forest"
}

# Make the POST request to the predict endpoint
try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # Raise an HTTPError for bad responses
except requests.exceptions.HTTPError as err:
    print(f"Error: {err}")
else:
    # Check the response status and print the results
    if response.status_code == 200:
        results = response.json()
        for idx, result in enumerate(results, start=1):
            print(f"{idx}. Sequence: {result['sequence']} - Result: {result['result']} - Probability: {result['probability']:.2f}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
