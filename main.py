#!/usr/bin/env python3
# main.py

import numpy as np
import cv2
import time
from tqdm import tqdm

# other people did this differently, whats that about? :(
def dense_optical_flow(prvs, frame, mask):
  flow = cv2.calcOpticalFlowFarneback(prvs,frame, None, 0.5, 3, 15, 3, 5, 1.5, 0) # I need to read about how this works lol
  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[...,1])

  mask[...,0] = angle * 180 / np.pi / 2 # no clue what this means too
  mask[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
  rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

  return rgb 

# code from main() is repeated here, it would be cool if I were to fix this TODO
def process_vid(vid_location, labels_location):
  cap = cv2.VideoCapture(vid_location)
  labels = open(labels_location, "r")
  processed = [] # kind of

  # get first frame for optical flow
  ret, first_frame = cap.read()
  prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
  mask = np.zeros_like(first_frame)
  mask[..., 1] = 255 # no clue what the mask is for (possibly to draw onto?)
  
  # processed for ML (hopefully)
  for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    try:
      ret, frame = cap.read()
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      rgb = dense_optical_flow(prev_frame, gray_frame, mask)   
      rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) # this is really not that efficient
      processed.append([np.array(rgb), labels.readline()])
    except:
      print("something went wrong")

  # close stuff
  cap.release() # cv VideoCapture  
  labels.close() # file
  return processed

def main():
  # get rid of the dumb UI elements we don't need
  cv2.namedWindow('input', cv2.WINDOW_GUI_NORMAL) # removes the weird buttons
  cv2.resizeWindow("input", 1280, 480) # resize window
  cv2.moveWindow('input', -1500, 300) 
  cap = cv2.VideoCapture("data/train.mp4")

  # get first frame for optical flow
  ret, first_frame = cap.read()
  prev_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
  mask = np.zeros_like(first_frame)
  mask[..., 1] = 255 # no clue what the mask is for (possibly to draw onto?)

  # main loop
  while (cap.isOpened()):
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = dense_optical_flow(prev_frame, gray_frame, mask)
    
    # join the two imgs to be fancy :)
    img_concate = np.concatenate((frame, rgb),axis=1)
    cv2.imshow('input', img_concate)
    
    prev_frame = gray_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
      exit()
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  print("PRESS 'q' TO EXIT BRO")
  main()

