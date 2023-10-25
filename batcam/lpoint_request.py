import requests
import json
import base64

USER = "admin"
PASSWORD = "admin"
CAMERA_IP = "192.168.2.2"
credential = f"{USER}:{PASSWORD}"
base64EncodedAuth = base64.b64encode(credential.encode()).decode()

API_HOST = f"http://{CAMERA_IP}"
"""
Create address for Camera REST API.
"""

url = API_HOST + "/beamforming/setting"
"""
Camera beamforming settings path for setting `index_l_channel`, `index_l`
"""


headers = {
    "Authorization": f"Basic {base64EncodedAuth}"
}
"""
The HTTP REST Header value.

Camera requires HTTP BasicAuth for authorization, so create base64 string and pass the value.
"""

camera_settings = {
    "low_cut": 2000,
    "high_cut": 45000,
    "gain": 100,
    "distance": 10,
    "x_cal": -0.02,
    "y_cal": 0.02,
    "index_l_channel": 0, # This could be 0 or 1
    "index_l": "1199,0"
    # Two points of Listening Point, 
    # if you send index_l_channel to 0, you can get sound of 20th point,
    # if you send index_l_channel to 1, you can get sound of 580th point.
}
"""
The actual body of camera's beamforming settings.
"""

print("Requesting body: " + str(camera_settings) + "\n")
    
try:
    current_setting = requests.get(url, headers=headers)
    print("Current Camera setting: " + str(current_setting.json()) + "\n")

    response = requests.patch(url, headers=headers, data=camera_settings)
    print("Result: " + str(response.json()) + "\n")
except Exception as ex:
    print(ex)