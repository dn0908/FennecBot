"""
* websocket: Module that required for connecting websocket.
* base64: The module for creating base64 string using in Basic Auth from websocket.
* rel: The scheduler for dispatching websocket
* json: The module for decoding json message from websocket.
"""
import base64
import json
import rel
import websocket
from interpolator import interpolate, plt_result


USER = "user"
PASSWORD = "user"
CAMERA_IP = "192.168.1.3"
credential = f"{USER}:{PASSWORD}"
base64EncodedAuth = base64.b64encode(credential.encode()).decode()


def on_message(_, message):
    dict_ = json.loads(message)
    gain = dict_["gain"]
    bf_data = list(dict_["bf"])
    cvted_data = interpolate(bf_data, gain, 3)
    plt_result(cvted_data)


def on_error(_, error):
    print(error)


def on_close(_, close_status_code, close_msg):
    print("### closed ###")


def on_open(socket):
    print("Opened connection")
    message = '{"type" : "subscribe", "id" : 0}'
    socket.send(message)
    print("Message sent")


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        f"ws://{CAMERA_IP}:80/ws",
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
