import cv2
import depth
import filters
from managers import WindowManager, CaptureManager
import rects
from trackers import FaceTracker

class Cameo(object):
    
    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                             self.onKeypress)
        self._captureManager = CaptureManager(
            cv2.VideoCapture(0), self._windowManager, True)
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False
        self._curveFilter = filters.BGRPortraCurveFilter()
    
    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            if frame is not None:
                
                self._faceTracker.update(frame)
                faces = self._faceTracker.faces
                rects.swapRects(frame, frame,
                                [face.faceRect for face in faces])
            
                filters.strokeEdges(frame, frame)
                self._curveFilter.apply(frame, frame)
                
                if self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame)
            
            self._captureManager.exitFrame()
            self._windowManager.processEvents()
    
    def onKeypress(self, keycode):
        """Handle a keypress.
        
        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        x      -> Start/stop drawing debug rectangles around faces.
        escape -> Quit.
        
        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    'screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: # x
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

class CameoDouble(Cameo):
    
    def __init__(self):
        Cameo.__init__(self)
        self._hiddenCaptureManager = CaptureManager(
            cv2.VideoCapture(1))
    
    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            self._hiddenCaptureManager.enterFrame()
            frame = self._captureManager.frame
            hiddenFrame = self._hiddenCaptureManager.frame
            
            if frame is not None:
                if hiddenFrame is not None:
                    self._faceTracker.update(hiddenFrame)
                    hiddenFaces = self._faceTracker.faces
                    self._faceTracker.update(frame)
                    faces = self._faceTracker.faces
                
                    i = 0
                    while i < len(faces) and i < len(hiddenFaces):
                        rects.copyRect(
                            hiddenFrame, frame, hiddenFaces[i].faceRect,
                            faces[i].faceRect)
                        i += 1
                
                filters.strokeEdges(frame, frame)
                self._curveFilter.apply(frame, frame)
                
                if hiddenFrame is not None and self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame)
            
            self._captureManager.exitFrame()
            self._hiddenCaptureManager.exitFrame()
            self._windowManager.processEvents()

class CameoDepth(Cameo):
    
    def __init__(self):
        self._windowManager = WindowManager('Cameo',
                                             self.onKeypress)
        device = cv2.CAP_OPENNI # uncomment for Microsoft Kinect via OpenNI
        #device = cv2.CAP_OPENNI_ASUS # uncomment for Asus Xtion via OpenNI
        #device = cv2.CAP_OPENNI2 # uncomment for Microsoft Kinect via OpenNI2
        #device = cv2.CAP_OPENNI2_ASUS # uncomment for Asus Xtion via OpenNI2
        self._captureManager = CaptureManager(
            cv2.VideoCapture(device), self._windowManager, True)
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False
        self._curveFilter = filters.BGRPortraCurveFilter()
    
    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            self._captureManager.channel = cv2.CAP_OPENNI_DISPARITY_MAP
            disparityMap = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_VALID_DEPTH_MASK
            validDepthMask = self._captureManager.frame
            self._captureManager.channel = cv2.CAP_OPENNI_BGR_IMAGE
            frame = self._captureManager.frame
            
            if frame is not None:
                self._faceTracker.update(frame)
                faces = self._faceTracker.faces
                masks = [
                    depth.createMedianMask(
                        disparityMap, validDepthMask, face.faceRect) \
                    for face in faces
                ]
                rects.swapRects(frame, frame,
                                [face.faceRect for face in faces], masks)
                
                filters.strokeEdges(frame, frame)
                self._curveFilter.apply(frame, frame)
                
                if self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame)
                
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

if __name__=="__main__":
    #Cameo().run() # uncomment for single camera
    CameoDouble().run() # uncomment for double camera
    #CameoDepth().run() # uncomment for depth camera
