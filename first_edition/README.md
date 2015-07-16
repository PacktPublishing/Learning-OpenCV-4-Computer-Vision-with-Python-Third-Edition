# Learn OpenCV with Python (First Edition Code)

This is the code for the first edition of the book.
All contained in this folder was written by Joseph Howse and can be found in its original form 
at [http://nummist.com/opencv/#book.3923]

I (Joe Minichino - techfort) only updated the code so it is compatible with OpenCV 3, which mostly consisted of the following changes:

1. replace all `cv2.cv.` occurrences with `cv2.`
2. replace all call to `VideoWriter.retrieve` to take no arguments (channel keyword not supported)
3. replace all calls to `cv2.Color` with `cv2.cvtColor`
4. replace all calls to `cv2.CV_FOURCC` with `cv2.VideoWriter_fourcc`
5. replace the flag `cv2.CV_HAAR_SCALE_IMAGE` with `cv2.CASCADE_SCALE_IMAGE`
