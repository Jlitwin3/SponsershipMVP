import requests
try:
    r = requests.get("http://localhost:5001/api/status")
    print(f"Status Code: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Error: {e}")
