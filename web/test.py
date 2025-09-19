import requests

url = "https://hrpolicyragv2-crcpdkcudnbucwbx.uksouth-01.azurewebsites.net/chat"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "dev-secret"
}
payload = {
    "input": "What is our sick leave policy?",
    "session_id": "rich"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())