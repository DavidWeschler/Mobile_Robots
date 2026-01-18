import socket
import struct
import time

# Replace with your NXT's Bluetooth address
# You can find it in Windows Settings > Bluetooth
NXT_ADDRESS = "00:16:53:0A:A0:2D"  # Device: aviam
NXT_PORT = 1  # RFCOMM port

# Connect to NXT by Bluetooth using native Windows socket
sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
    sock.connect((NXT_ADDRESS, NXT_PORT))
except OSError as e:
    print(f"Connection error: {e}")
    print("Make sure your NXT is turned on and Bluetooth is paired.")
    print("If error persists, try turning Bluetooth off and on in Windows.")
    sock.close()
    exit(1)

# NXT message write command
# Direct command: message write (mailbox 0)
def message_write(mailbox, message):
    # NXT Direct Command: MessageWrite (0x09)
    msg_bytes = message.encode() + b'\x00'  # null-terminated
    cmd = bytes([0x00, 0x09, mailbox, len(msg_bytes)]) + msg_bytes
    # Add length header (little endian)
    packet = struct.pack('<H', len(cmd)) + cmd
    sock.send(packet)
    # Read response
    response = sock.recv(64)
    return response

# Continuous input loop
print("Connected! Type messages to send to NXT (Ctrl+C to exit):")
try:
    while True:
        msg = input("> ")
        if msg:  # Only send if not empty
            response = message_write(1, msg)  # Mailbox 1 to match NXC code
            print(f"Sent: {msg}")
except KeyboardInterrupt:
    print("\nDisconnecting...")
finally:
    sock.close()
    print("Connection closed.")