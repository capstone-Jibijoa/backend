import requests
import json

url = "http://127.0.0.1:8000/api/search"
payload = {
    "query": "서울 30대 IT 직장인 100명",
    "search_mode": "all"
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
    
    print("--- 요청 성공 ---")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
    
except requests.exceptions.HTTPError as err:
    print(f"HTTP Error: {err}")
except Exception as e:
    print(f"An error occurred: {e}")