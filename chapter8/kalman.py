import cv2, numpy as np

measurements=[]
predictions=[]
frame = np.zeros((400,400,3), np.uint8) # the empty frame working as a canvas
current_measurement = np.array((2,1), np.float32) # measurement array 
current_prediction = np.zeros((2,1), np.float32) # tracked / prediction

def onmouse(event, x, y, s, p):
    global current_measurement, measurements
    current_measurement = np.array([[np.float32(x)],[np.float32(y)]])
    measurements.append((x,y))

def draw(frame, measurements, predictions):
    for i in range(len(measurements)-1):
        cv2.line(frame, measurements[i], measurements[i+1], (0,100,0))

    for i in range(len(predictions)-1):
        cv2.line(frame, predictions[i], predictions[i+1], (0,0,200))

cv2.namedWindow("kalman")
cv2.setMouseCallback("kalman",onmouse);

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

while True:
    kalman.correct(current_measurement)
    current_prediction = kalman.predict()
    predictions.append((int(current_prediction[0]),int(current_prediction[1])))
    draw(frame, measurements, predictions)
    cv2.imshow("kalman",frame)
    k = cv2.waitKey(30) &0xFF
    if k == 27: break

cv2.destroyAllWindows()
