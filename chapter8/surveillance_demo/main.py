"""
  Surveillance Demo: Tracking Pedestrians in Camera Feed

  The application opens a video (could be a camera or a video file)
  and tracks pedestrians in the video.
"""

import cv2
import numpy as np
import os.path as path

"""
each pedestrian is composed of a ROI, an ID and a Kalman filter
so we create a Pedestrian class to hold the object state
"""
class Pedestrian():
  def __init__(self, id, frame, track_window):
    # set up the roi
    self.id = int(id)
    x,y,w,h = track_window
    self.roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    
    roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
    self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # set up the kalman
    self.kalman = cv2.KalmanFilter(4,2)
    self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    self.measurement = np.array((2,1), np.float32) 
    self.prediction = np.zeros((2,1), np.float32)

  def update(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0], self.roi_hist,[0,180],1)
    
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    (cx, cy), radius = cv2.minEnclosingCircle(pts)
    kalman.correct(center(pts))
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    prediction = kalman.predict()
    cv2.circle(frame, (prediction[0], prediction[1]), int(radius), (0, 255, 0))
    cv2.imshow('img2',img2)

    print "updating %d" % self.id



def main():
  camera = cv2.VideoCapture(path.join(path.dirname(__file__), "768x576.avi"))
  ret, frame = camera.read()
  counter = 0
  pedestrians = {}
  if (ret is False):
    print "failed to read frame... exiting."
    return
  else:
      ret, frame = camera.read()
      fgmask = bs.apply(frame)
      th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
      th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 2)
      dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,3)), iterations = 2)
      image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      for c in contours:
        if cv2.contourArea(c) > 800:
          tw = (x,y,w,h) = cv2.boundingRect(c)
          #cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)
          pedestrians[counter] = Pedestrian(counter, frame, tw)
          counter += 1


  while True:
    grabbed, frame = camera.read()
    pedstrian.update(frame)
    cv2.imshow("video", frame)
    if cv2.waitKey(1000 / 10) >= 0:
      break

if __name__ == "__main__":
  main()
