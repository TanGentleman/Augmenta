import requests
from pyperclip import copy, paste

IP = "http://127.0.0.1"
PORT = 8000
input_text = paste().strip()
criteria = "The language is English."

response = requests.post(f"{IP}:{PORT}/", params={"text": input_text, "criteria": criteria})
# Rewrite as a faster terminal command
response_string = response.text

print(response_string)
copy(response_string)
