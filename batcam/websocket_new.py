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

data_list = []

# def save_wav(timestamp, audio_data):
#     audio_data = np.asarray(audio_data)
#     # 60 * 64 크기의 frame에서 각 포인트별 음성 데이터 추출
#     frame = np.random.rand(60 * 64)

#     # wave 파일 생성
#     with wave.open(f"{timestamp}.wav", 'w') as wav:
#         # 채널 수, 샘플 폭, 샘플링 빈도 설정
#         wav.setparams((1, 2, 96000, 60 * 64, 'NONE', 'not compressed'))
#         # 60개의 frame에서 각 포인트별 음성 데이터 추출 후 wav 파일에 쓰기
#         for i in range(60):
#             audio_data = frame[i * 64:(i + 1) * 64]
#             wav.writeframes(audio_data)

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
    
    data_= json.loads(message)
    timestamp = data_["timestamp"]
    l_point_0 = list(data_["l_point_0"])
    # l_point_1 = list(data_["l_point_1"])


    count_num += 1
    print("count num : ",count_num)

    if count_num >= 16 and count_num < 32:
        data_list.append(l_point_0)

    if count_num == 32:
        # save_wav(timestamp, data_list)
        df = pd.DataFrame(data_list, index=None, columns=None)
        save_csv(timestamp, df)
        print(np.asarray(data_list).shape)
        print('csv saved')
        # Unsubscribe and close the connection
        unsubscribe_msg = json.dumps({"type": "unsubscribe", "id": trigger_id})
        ws.send(unsubscribe_msg)
        ws.close()


    # if data.get("event_id") == 3:
    #     timestamp = data["timestamp"]
    #     audio_data = data["1_point_0"] # For websocket audio (ID:2)
    #     time.sleep(5)
    #     save_wav(timestamp, audio_data)
    #     print('wav saved')

        # # Record 5 seconds of audio
        # time.sleep(5)
        # # Unsubscribe and close the connection
        # unsubscribe_msg = json.dumps({"type": "unsubscribe", "id": trigger_id})
        # ws.send(unsubscribe_msg)
        # ws.close()
    # print(data)
    # Save data to file
    # with open("data.txt", "a") as f:
    #     f.write(json.dumps(data) + "\n")

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
    print(count_num)
    ws = websocket.WebSocketApp(f"ws://{BATCAM_IP}/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                subprotocols=["subscribe"],
                                header={"Authorization": f"Basic %s" % base64EncodedAuth}
                            )
    ws.run_forever(dispatcher=rel, reconnect=5)
    rel.signal(2, rel.abort)
    rel.dispatch()
