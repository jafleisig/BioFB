import serial
import time

print("begin")

ser = serial.Serial('COM4')

while True:
    codein = str(input('Enter input: '))
    ser.write(codein.encode('utf-8'))


