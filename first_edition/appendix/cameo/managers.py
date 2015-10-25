import cv2
import numpy
import pygame
import time
import utils


class CaptureManager(object):
    
    def __init__(self, capture, previewWindowManager = None,
                 shouldMirrorPreview = False):
        
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        
        self._startTime = None
        self._framesElapsed = long(0)
        self._fpsEstimate = None
    
    @property
    def channel(self):
        return self._channel
    
    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None
    
    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            # As of OpenCV 3.0, VideoCapture.retrieve() no longer supports
            # the channel argument.
            # _, self._frame = self._capture.retrieve(channel = self.channel)
            _, self._frame = self._capture.retrieve()
        return self._frame
    
    @property
    def isWritingImage(self):
        return self._imageFilename is not None
    
    @property
    def isWritingVideo(self):
        return self._videoFilename is not None
    
    def enterFrame(self):
        """Capture the next frame, if any."""
        
        # But first, check that any previous frame was exited.
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'
        
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()
    
    def exitFrame(self):
        """Draw to the window. Write to files. Release the frame."""
        
        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return
        
        # Update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate =  self._framesElapsed / timeElapsed
        self._framesElapsed += 1
        
        # Draw to the window, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)
        
        # Write to the image file, if any.
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None
        
        # Write to the video file, if any.
        self._writeVideoFrame()
        
        # Release the frame.
        self._frame = None
        self._enteredFrame = False
    
    def writeImage(self, filename):
        """Write the next exited frame to an image file."""
        self._imageFilename = filename
    
    def startWritingVideo(
            self, filename,
            encoding = cv2.VideoWriter_fourcc('M','J','P','G')):
        """Start writing exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encoding
    
    def stopWritingVideo(self):
        """Stop writing exited frames to a video file."""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
    
    def _writeVideoFrame(self):
        
        if not self.isWritingVideo:
            return
        
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(
                        cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(
                        cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename, self._videoEncoding,
                fps, size)
        
        self._videoWriter.write(self._frame)


class WindowManager(object):
    
    def __init__(self, windowName, keypressCallback = None):
        self.keypressCallback = keypressCallback
        
        self._windowName = windowName
        self._isWindowCreated = False
    
    @property
    def isWindowCreated(self):
        return self._isWindowCreated
    
    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True
    
    def show(self, frame):
        cv2.imshow(self._windowName, frame)
    
    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False
    
    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            # Discard any non-ASCII info encoded by GTK.
            keycode &= 0xFF
            self.keypressCallback(keycode)

class PygameWindowManager(WindowManager):
    
    def createWindow(self):
        pygame.display.init()
        pygame.display.set_caption(self._windowName)
        self._isWindowCreated = True
    
    def show(self, frame):
        
        # Find the frame's dimensions in (w, h) format.
        frameSize = frame.shape[1::-1]
        
        # Convert the frame to RGB, which Pygame requires.
        if utils.isGray(frame):
            conversionType = cv2.COLOR_GRAY2RGB
        else:
            conversionType = cv2.COLOR_BGR2RGB
        rgbFrame = cv2.cvtColor(frame, conversionType)
        
        # Convert the frame to Pygame's Surface type.
        pygameFrame = pygame.image.frombuffer(
            rgbFrame.tostring(), frameSize, 'RGB')
        
        # Resize the window to match the frame.
        displaySurface = pygame.display.set_mode(frameSize)
        
        # Blit and display the frame.
        displaySurface.blit(pygameFrame, (0, 0))
        pygame.display.flip()
    
    def destroyWindow(self):
        pygame.display.quit()
        self._isWindowCreated = False
    
    def processEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and \
                    self.keypressCallback is not None:
                self.keypressCallback(event.key)
            elif event.type == pygame.QUIT:
                self.destroyWindow()
                return
