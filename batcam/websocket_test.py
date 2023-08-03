import websocket
import json
import threading

import wave
import time

import base64
# Use code blocks to display formatted content such as code
import rel
import pickle
import pandas as pd
import csv
import numpy as np

USER = "user"
PASSWORD = "user"
BATCAM_IP = "192.168.2.2"
credential = f"{USER}:{PASSWORD}"
base64EncodedAuth = base64.b64encode(credential.encode()).decode()

trigger_id = None
count_num = 0
save_num = 0

data_list = []


# Create threading event when to close connection
exit_event = threading.Event()


def save_csv(timestamp, audio_data):
    # Create a csv writer object to write the csv file
    filename = f"{timestamp}.csv"
    csv_file = open(filename, "w")
    csv_writer = csv.writer(csv_file)
    df = pd.DataFrame(data_list)
    df.to_csv(filename, index = False)

def on_message(_, message):
    global trigger_id
    global count_num
    global save_num
    
    data_= json.loads(message)
    timestamp = data_["timestamp"]
    l_point_0 = list(data_["l_point_0"])
    # l_point_1 = list(data_["l_point_1"])

    count_num += 1
    print("count num : ",count_num)

    if count_num >= 16 and count_num < 32:
        data_list.append(l_point_0)

    if count_num == 32:
        df = pd.DataFrame(data_list, index=None, columns=None)
        save_csv('l_point_data', df)
        print('csv saved')

def on_error(_, error):
    print(error)

def on_close(_, close_status_code, close_msg):
    print("### closed ###")
    # Set exit_event to signal closed
    exit_event.set()

'''
|  EVENT ID  |      DATA      |                   json form                     |
---------------------------------------------------------------------------------
|     0      |   beamforming  |          event_id, bf, gain, timestamp          |
|     1      |     decibel    |           event_id, timestamp, decibel          |
|     2      |    ws audio    |          timestamp, event_id, gain, ws          |
|     3      |  1 point audio | timestamp, event_id, gain, 1_point_0, 1_point_1 |
'''

def on_open(socket):
    print("Opened connection")
    message = '{"type" : "subscribe", "id" : 3}'
    socket.send(message)
    print("Message sent")

def websocket_thread():
    # Create a new WebSocket connection
    ws = websocket.WebSocketApp(f"ws://{BATCAM_IP}/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                subprotocols=["subscribe"],
                                header={"Authorization": f"Basic %s" % base64EncodedAuth}
                            )
    # Run the WebSocket connection until the exit_event is set
    ws.run_forever(dispatcher=rel, ping_interval=5, ping_timeout=2, close_timeout=1, http_proxy_host=None, http_proxy_port=None, sslopt=None, suppress_origin=False, ping_payload=b'ping')



# Start the WebSocket connection in a new thread
websocket_thread = threading.Thread(target=websocket_thread)
websocket_thread.start()

# Wait for the WebSocket connection to finish (i.e., when the exit_event is set)
websocket_thread.join()

# Save CSV
save_csv('l_point_data', data_list)
print('csv saved')