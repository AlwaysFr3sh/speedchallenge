#!/usr/bin/env python3
# main.py

import numpy as np
import cv2
import time
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils import SpeedNet
from utils import downscale

# other people did this differently, whats that about? :(
def dense_optical_flow(prvs, frame, mask):
  flow = cv2.calcOpticalFlowFarneback(prvs,frame, None, 0.5, 3, 15, 3, 5, 1.5, 0) 
  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[...,1])

  mask[...,0] = angle * 180 / np.pi / 2 # no clue what this means too
  mask[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
  rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

  return rgb 

def get_predicted_speed(net, image):
  with torch.no_grad():
    net_out = net(image.view(-1, 1, 240, 320))
  return float(net_out[0][0])

# more repeated code, which is bad, TODO: fix
def average(lst):
  return sum(lst) / len(lst)

# returns grayscale image
def grayscale(frame):
  return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def main():
  # get rid of the dumb UI elements we don't need
  cv2.namedWindow('input', cv2.WINDOW_GUI_NORMAL) # removes the weird buttons
  cv2.resizeWindow("input", 640, 480) # resize window
  cv2.moveWindow('input', -1500, 300) 
  cap = cv2.VideoCapture("data/train.mp4")

  # get first frame for optical flow
  ret, first_frame = cap.read()
  first_frame = downscale(first_frame, 50)
  prev_frame = grayscale(first_frame) 
  mask = np.zeros_like(first_frame)
  mask[..., 1] = 255 # no clue what the mask is for (possibly to draw onto?)

  # get speed labels
  # TODO: make sure speed labels align properly in training
  # I would do this by checking that every frame has a corresponding speed, and the last frame speed
  # isn't empty (I'm sure its fine)
  real_speeds = open("data/train.txt", 'r')
  net = SpeedNet()
  net.load_state_dict(torch.load("models/big_data_model.pth"))
  acc = [] # accumulated speed errors for mse

  # main loop
  while (cap.isOpened()):
    ret, frame = cap.read()
    gray_frame = grayscale(downscale(frame, 50))
    rgb = dense_optical_flow(prev_frame, gray_frame, mask)
    rgb = grayscale(rgb)
    rgb = torch.Tensor(rgb)

    # write speeds onto the image
    speed = real_speeds.readline()[:-1] #chop whatever the unknown traling char is
    pred_speed = get_predicted_speed(net, rgb)
    acc.append((float(speed) - pred_speed)**2)

    # write stats to frame before display
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, speed ,(0,30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, str(pred_speed), (0, 60), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(average(acc)), (0, 90), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    
    # display
    cv2.imshow('input', frame)

    prev_frame = gray_frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
      exit()
      break

  print("MSE", average(acc))
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  print("PRESS 'q' TO EXIT BRO")
  main()

