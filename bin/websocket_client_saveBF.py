# Use code blocks to display formatted content such as code
import base64
import json
import rel
import websocket
from interpolator import interpolate, plt_result
import pickle
import pandas as pd

import csv

# Define some constants
USER = "user"
PASSWORD = "user"
CAMERA_IP = "192.168.2.2"

# Create a file object to write the data
data_file = open("data.pkl", "wb")

data_list = []

# Create a csv writer object to write the csv file
csv_file = open("data.csv", "w")
csv_writer = csv.writer(csv_file)

count_num = 0


def on_message(_, message):
    global count_num
    # Parse the message as a JSON object
    dict_ = json.loads(message)
    gain = dict_["gain"]
    bf_data = list(dict_["bf"])

    # Interpolate and plot the data
    cvted_data = interpolate(bf_data, gain, 3)
    plt_result(cvted_data)

    # Write the data to the pickle file
    #pickle.dump(cvted_data, data_file)
    # Write the data to the csv file
    #csv_writer.writerow(cvted_data)
    
    count_num += 1
    print(count_num)
    
    if count_num >= 10 and count_num < 20:
        cvted_data = cvted_data.reshape(-1)
        print(cvted_data.shape)
        data_list.append(cvted_data)
    	
    if count_num == 20:
        df = pd.DataFrame(data_list)
        file_name = f'data.csv'
        df.to_csv(file_name, index = False)
        print(f"Saved {file_name}")
        count_num = 0
        print("count num to zero")

    
    #for i in range (0, n_rows, time_step):
        #chunk = df.iloc[i:i+time_step]
        #file_name = f"data_{i}.csv"
        #chunk.to_csv(file_name, index = False)
        # print a message to confirm saving
        #print(f"Saved {file_name}")


def on_error(_, error):
    print(error)

def on_close(_, close_status_code, close_msg):
    print("### closed ###")
    # Close the files after the websocket is closed
    data_file.close()
    csv_file.close()

def on_open(socket):
    print("Opened connection")
    message = '{"type" : "subscribe", "id" : 0}'
    socket.send(message)
    print("Message sent")

if __name__ == "__main__":
    # Create a websocket object with authentication and callbacks
    count_num = 0
    ws = websocket.WebSocketApp(
        f"ws://{CAMERA_IP}:80/ws",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        subprotocols=["subscribe"],
        header={"Authorization": f"Basic %s" % base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()}
    )
    # Run the websocket until interrupted
    ws.run_forever(dispatcher=rel, reconnect=5)
    rel.signal(2, rel.abort)
    rel.dispatch()
