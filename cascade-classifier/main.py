import numpy as np
import cv2
import os

a = np.random.randint(0,255,(100,100,3)).astype(np.uint8)
print(os.getcwd())
while True:
    cv2.imshow('test',a)
    if cv2.waitKey(1) == ord('q'):
        break