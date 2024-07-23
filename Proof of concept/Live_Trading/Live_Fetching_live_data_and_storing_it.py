# Get Tick Data 'Use Websocket'
from kiteconnect import KiteTicker
user_id = kite.profile()["user_id"]
kws = KiteTicker(api_key="TradeViaPython", access_token=enctoken+"&user_id="+user_id)

class DataCollector:
    def __init__(self):
        self.data = []
        
    def store_ticks(self, ticks):
        self.data.append(ticks)

obj = DataCollector()

def on_ticks(ws, ticks):
    obj.store_ticks(ticks)
    print("Stored : ", ticks)

def on_connect(ws, response):
    print("WebSocket: Connected")
    instrument_tokens = [779521]  # Replace with desired instrument tokens
    ws.subscribe(instrument_tokens)
    ws.set_mode(ws.MODE_QUOTE, instrument_tokens)

def on_close(ws, code, reason):
    print(f"WebSocket: Closed (Code: {code}, Reason: {reason})")
    ws.stop()

def on_error(ws, code, reason):
    print(f"WebSocket: Error (Code: {code}, Reason: {reason})")

def on_noreconnect(ws):
    print("WebSocket: No reconnect will be attempted")

def on_reconnect(ws, attempts_count):
    print(f"WebSocket: Reconnecting... Attempt: {attempts_count}")

# Set the callbacks
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close
kws.on_error = on_error
kws.on_noreconnect = on_noreconnect
kws.on_reconnect = on_reconnect

# Connect to WebSocket
kws.connect(threaded=True)

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Keyboard Interrupt: Stopping WebSocket")
    kws.unsubscribe([779521])
    kws.close()
