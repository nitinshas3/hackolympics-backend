import serial

ser = serial.Serial("COM7", 9600)

while True:
    print(ser.readline())