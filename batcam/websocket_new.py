import websocket
import json
import threading

import wave
import time

import base64


USER = "user"
PASSWORD = "user"
BATCAM_IP = "192.168.1.3"
credential = f"{USER}:{PASSWORD}"
base64EncodedAuth = base64.b64encode(credential.encode()).decode()

trigger_id = None

def save_wav(timestamp, audio_data):
    with wave.open(f"{timestamp}.wav", "wb") as wav_file:
        # Set parameters of WAV file
        wav_file.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        # Write audio data to file
        wav_file.writeframes(audio_data)

def on_message(ws, message):
    global trigger_id
    data = json.loads(message)
    if data.get("event_id") == 3:
        timestamp = data["timestamp"]
        audio_data = data["1_point_0"] # For websocket audio (ID:2)
        save_wav(timestamp, audio_data)
        print('wav saved')

        # # Record 5 seconds of audio
        # time.sleep(5)
        # # Unsubscribe and close the connection
        # unsubscribe_msg = json.dumps({"type": "unsubscribe", "id": trigger_id})
        # ws.send(unsubscribe_msg)
        # ws.close()
    print(data)
    # Save data to file
    with open("data.txt", "a") as f:
        f.write(json.dumps(data) + "\n")

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

'''
|  EVENT ID  |      DATA      |                   json form                     |
---------------------------------------------------------------------------------
|     0      |   beamforming  |          event_id, bf, gain, timestamp          |
|     1      |     decibel    |           event_id, timestamp, decibel          |
|     2      |    ws audio    |          timestamp, event_id, gain, ws          |
|     3      |  1 point audio | timestamp, event_id, gain, 1_point_0, 1_point_1 |
'''

def on_open(ws):
    def run(*args):
        trigger_id = 3 # for 1 point audio
        subscribe_msg = json.dumps({"type": "subscribe", "id": trigger_id})
        ws.send(subscribe_msg)
        while True:
            user_input = input()
            if user_input == "exitFX":
                unsubscribe_msg = json.dumps({"type": "unsubscribe", "id": trigger_id})
                ws.send(unsubscribe_msg)
                ws.close()
                break
    thread = threading.Thread(target=run)
    thread.start()

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(f"ws://{BATCAM_IP}/ws",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                subprotocols=["subscribe"],
                                header={"Authorization": f"Basic %s" % base64EncodedAuth}
                            )
    ws.on_open = on_open
    ws.run_forever()
