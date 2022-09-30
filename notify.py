import requests
import os
path = 'key.txt'

def lineNotify(message):
    payload = {'message':message}
    return _lineNotify(payload)

def _lineNotify(payload,file=None):
    url = 'https://notify-api.line.me/api/notify'
    token = "Your Token"
    headers = {'Authorization':'Bearer ' + token}
    return requests.post(url,headers = headers, data = payload, files = file)

def read_token(path):
    try:
        with open(path, 'r') as f:
            print(f.read().strip())
            return f"{f.read().strip()}"
    except FileNotFoundError:
        print(f"Please type your path again!!! old:{path}")


