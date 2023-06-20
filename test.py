import requests

BASE = 'http://127.0.0.1:5000/'

num_pred = input('How many predictions would you like to see?')
response = requests.get(BASE + f"predict/{num_pred}")
print(response.json())