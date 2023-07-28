import base64
import json
import rel
import websocket
import wave
import time
# from interpolator import interpolate, plt_result


USER = "user"
PASSWORD = "user"
BATCAM_IP = "192.168.1.3"
credential = f"{USER}:{PASSWORD}"
base64EncodedAuth = base64.b64encode(credential.encode()).decode()

def save_wav(filename, audio_data):
    with wave.open(f"{filename}.wav", "wb") as wav_file:
        # Set parameters of WAV file
        wav_file.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        # Write audio data to file
        wav_file.writeframes(audio_data)

def on_message(_, message):
    dict_ = json.loads(message)
    print(message)
    gain = dict_["gain"]
    timestamp = dict_["timestanp"]
    audio_0 = list(dict_["1_point_0"])
    audio_1 = list(dict_["1_point_1"])
    print(audio_1)
    time.sleep(5)
    save_wav(timestamp+'_audio0', audio_0)
    print('wav0 saved')
    save_wav(timestamp+'_audio0', audio_1)
    print('wav0 saved')


def on_error(_, error):
    print(error)


def on_close(_, close_status_code, close_msg):
    print("### closed ###")


def on_open(socket):
    print("Opened connection")
    message = '{"type" : "subscribe", "id" : 3}'
    socket.send(message)
    print("Message sent")


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        f"ws://{BATCAM_IP}/ws",
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
