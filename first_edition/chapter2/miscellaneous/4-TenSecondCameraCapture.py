import cv2

cameraCapture = cv2.VideoCapture(0)
fps = 30 # an assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter(
    'MyOutputVid.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

success, frame = cameraCapture.read()
numFramesRemaining = 10 * fps - 1
while numFramesRemaining > 0:
    if frame is not None:
        videoWriter.write(frame)
    success, frame = cameraCapture.read()
    numFramesRemaining -= 1
