import serial
import time

print("begin")

ser = serial.Serial('COM4', 9800, timeout = 1)

while True:
    print("going")
    ser.write(b'H')
    time.sleep(1)


