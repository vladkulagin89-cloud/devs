import requests
import json

def make_curl_request(url):
    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "Always answer in rhymes."
            },
            {
                "role": "user",
                "content": "Introduce yourself."
            }
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": True
    }

    response = requests.post(url, headers=headers, json=data, stream=True)
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

url = "http://192.168.31.124:1234/v1/chat/completions"
make_curl_request(url)


