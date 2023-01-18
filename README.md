


# Learning OpenCV 4 Computer Vision with Python 3 - Third Edition 

<a href="https://www.packtpub.com/data/learning-opencv-4-computer-vision-with-python-3-third-edition?utm_source=github&utm_medium=repository&utm_campaign=9781789531619"><img src="https://static.packt-cdn.com/products/9781789531619/cover/smaller" alt="Learning OpenCV 4 Computer Vision with Python 3 - Third Edition " height="256px" align="right"></a>

This is the code repository for [Learning OpenCV 4 Computer Vision with Python 3 - Third Edition](https://www.packtpub.com/data/learning-opencv-4-computer-vision-with-python-3-third-edition?utm_source=github&utm_medium=repository&utm_campaign=9781789531619), published by Packt.

**Get to grips with tools, techniques, and algorithms for computer vision and machine learning**

## What is this book about?
Computer vision is a rapidly evolving science, encompassing diverse applications and techniques. This book will not only help those who are getting started with computer vision but also experts in the domain. You’ll be able to put theory into practice by building apps with OpenCV 4 and Python 3.


This book covers the following exciting features:
* Install and familiarize yourself with OpenCV 4's Python 3 bindings 
* Understand image processing and video analysis basics 
* Use a depth camera to distinguish foreground and background regions 
* Detect and identify objects, and track their motion in videos 
* Train and use your own models to match images and classify objects 
* Detect and recognize faces, and classify their gender and age 
* Build an augmented reality application to track an image in 3D 
* Work with machine learning models, including SVMs, artificial neural networks(ANNs), and deep neural networks(DNNs)

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1789531616) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
import cv2

grayImage = cv2.imread('MyPic.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('MyPicGray.png', grayImage)
```

**Following is what you need for this book:**
If you are interested in learning computer vision, machine learning, and OpenCV in the context of practical real-world applications, then this book is for you. This OpenCV book will also be useful for anyone getting started with computer vision as well as experts who want to stay up-to-date with OpenCV 4 and Python 3. Although no prior knowledge of image processing, computer vision or machine learning is required, familiarity with basic Python programming is a must.

We provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781789531619_ColorImages.pdf).

With the following software and hardware list you can run all code files present in the book (Chapter 1-10).
### Software and Hardware List
| Chapter | Software required | OS required |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-10 | Python 3 (specifically, 3.5 or later) | Windows, Mac OS X, and Linux (Any) |
| 1-10 | OpenCV 4 | Windows, Mac OS X, and Linux (Any) |
| 1-10 | NumPy (any recent version) | Windows, Mac OS X, and Linux (Any) |
| 3 | SciPy (any recent version) | Windows, Mac OS X, and Linux (Any) |
| 4, 5 | OpenNI 2 | Windows, Mac OS X, and Linux (Any) |
| 4, 6 | Matplotlib (any recent version) | Windows, Mac OS X, and Linux (Any) |

#### Update about Optional Software Requirements
At the time the book was written, some of the code samples (in Chapters 6 and 7) depended on OpenCV's "non-free" modules in order to use the SIFT and SURF patented algorithms. Since then, the SIFT patent has expired and, starting in OpenCV 4.4.0, SIFT can be used without the "non-free" modules. There is now just one code sample (in Chapter 6) which depends on the "non-free" modules for SURF. If you wish to try the SURF sample, you will need to build OpenCV with the "non-free" modules from source (as per instructions in Chapter 1 for building from source). The pre-built `opencv-contrib-python-nonfree` pip package (also mentioned in Chapter 1) is no longer available.

### Related Products
* Hands-On Computer Vision with TensorFlow 2  [[Packt]](https://www.packtpub.com/application-development/hands-computer-vision-tensorflow-2?utm_source=github&utm_medium=repository&utm_campaign=9781788830645) [[Amazon]](https://www.amazon.com/dp/1788830644)
* OpenCV 4 for Secret Agents  [[Packt]](https://www.packtpub.com/product/opencv-4-for-secret-agents-second-edition/9781789345360) [[Amazon]](https://www.amazon.com/dp/1789345367)

## Get to Know the Authors
**Joseph Howse**
 lives in a Canadian fishing village with four cats; the cats like fish, but they prefer chicken.

Joseph provides computer vision expertise through his company, Nummist Media. His books include OpenCV 4 for Secret Agents, Learning OpenCV 4 Computer Vision with Python 3, OpenCV 3 Blueprints, Android Application Programming with OpenCV 3, iOS Application Development with OpenCV 3, and Python Game Programming by Example, published by Packt.

**Joe Minichino**
 is an R&D labs engineer at Teamwork. He is a passionate programmer who is immensely curious about programming languages and technologies and constantly experimenting with them. Born and raised in Varese, Lombardy, Italy, and coming from a humanistic background in philosophy (at Milan's Università Statale), Joe has lived in Cork, Ireland, since 2004. There, he became a computer science graduate at the Cork Institute of Technology.

## Troubleshooting and FAQ

### Issue: Camera input does not work on Windows

For some cameras and some versions of OpenCV, `cv2.VideoCapture` fails to capture camera input when it uses the Microsoft Media Foundation (MSMF) back-end. This issue may manifest itself with errors such as ``[ WARN:0@25.936] global C:\opencv\modules\videoio\src\cap_msmf.cpp (539) `anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback`` when you run camera input scripts such as `chapter02/5-CameraWindow.py`.

To work around the problem, define an environment variable with the name `OPENCV_VIDEOIO_PRIORITY_MSMF` and the value `0`. (You may need to reboot in order for global changes to your environment variables to take effect.) This change de-prioritizes OpenCV's MSMF back-end so that OpenCV will try to choose any other back-end, usually the Microsoft DirectShow back-end, which is more compatible.

Alternatively, to specify the preferred back-end in any given script, replace code such as `cv2.VideoCapture(0)` with code such as `cv2.VideoCapture(0, cv2.CAP_DSHOW)`, which specifies Microsoft DirectShow as the preferred back-end. However, be aware that such changes may reduce the portability of your code.

### Question: Will the sample code work with the upcoming OpenCV 5?

The sample code in this repository has been tested successfully with OpenCV's `5.x` development branch as of August 7, 2022. Thus, as far as we can tell at this point, it should work with the upcoming OpenCV 5 release.

A future edition of the book will provide more extensive coverage of new features in OpenCV 5. See [https://github.com/PacktPublishing/Learning-OpenCV-5-Computer-Vision-with-Python-Fourth-Edition](https://github.com/PacktPublishing/Learning-OpenCV-5-Computer-Vision-with-Python-Fourth-Edition).

## Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781789531619">https://packt.link/free-ebook/9781789531619 </a> </p>