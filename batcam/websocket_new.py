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

# def save_wav(timestamp, audio_data):
#     audio_data = np.asarray(audio_data)
#     audio_data = audio_data.reshape(-1)

#     # wave 파일 생성
#     with wave.open(f"{timestamp}.wav", 'w') as wav:
#         wav.setnchannels(1)
#         wav.setsampwidth(2)
#         wav.setframerate(sample_rate)
#         wav.writeframes(audio_data.astype(np.int16))


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

        # save_wav('l_point_data', data_list)
        # print('wav saved')
        

        # Unsubscribe and close the connection
        unsubscribe_msg = json.dumps({"type": "unsubscribe", "id": trigger_id})
        ws.send(unsubscribe_msg)
        ws.close()
        save_num = 1

def on_error(_, error):
    print(error)

def on_close(_, close_status_code, close_msg):
    print("### closed ###")
    # Close the files after the websocket is closed

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

if __name__ == "__main__":
    count_num = 0
    save_num = 0
    print(count_num)
    ws = websocket.WebSocketApp(f"ws://{BATCAM_IP}/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                subprotocols=["subscribe"],
                                header={"Authorization": f"Basic %s" % base64EncodedAuth}
                            )
    # time.sleep(10)
    # print('time passed')
    # ws.close()
    ws.run_forever(dispatcher=rel, reconnect=5)
    rel.signal(2, rel.abort)
    rel.dispatch()
    if save_num == 1:
        # Unsubscribe and close the connection
        unsubscribe_msg = json.dumps({"type": "unsubscribe", "id": trigger_id})
        ws.send(unsubscribe_msg)
        ws.close()
    # ws.run_forever(dispatcher=rel, reconnect=5)
    # rel.signal(2, rel.abort)
    # rel.dispatch()
    # ws.close()
