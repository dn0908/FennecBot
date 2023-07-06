# history 확인

import requests

BATCAM_IP = "YOUR_DEVICE_IP"
EVENT_ID = "YOUR_EVENT_ID"

url = f"http://{BATCAM_IP}/event/history?id={EVENT_ID}"
response = requests.get(url)

events = response.json()
print(events)