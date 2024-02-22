import pyautogui as pg
import sys
import time
import headMotionTracking as hmt

# keyboard_input = input("Enter a number: ")

dis = hmt.track_head_movement('shape_predictor_68_face_landmarks.dat')
print(dis)

if dis is not None:
    try:
        if dis > 50:
            pg.moveTo(445, 880, duration=2)
            time.sleep(2)
            pg.moveTo(400, 880, duration=2)
        else:
            print("no significant motion")
    except ValueError:
        print("error in detecting the head motion")
else:
    print("no motion detected")

# try:
#     value = int(inPut)
# except ValueError:
#     print("Please enter a valid number.")
# # if value exist:
#     pg.moveTo(1710, 734)
#     time.sleep(0.5)
#     pg.moveTo(1704, 775)
# else:
#     print("no input")