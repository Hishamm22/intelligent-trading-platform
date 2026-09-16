import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")

url = "https://testnet.binance.vision/api/v3/ping"
response = requests.get(url)

print("Status code:", response.status_code)
print("Response:", response.json())
print("API key loaded correctly:", api_key is not None)