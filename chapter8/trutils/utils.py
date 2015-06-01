import numpy as np

def suppress(boxes):
  

def area(box):
  return (box[2] - box[0]) * (box[3] - box[1])

def intersect(boxes, a, b):
  box1 = boxes[a]
  box2 = boxes[b]
  x = np.maximum(box1[0], box2[0])
  y = np.maximum(box1[1], box2[1])
  x2 = np.maximum(box1[2], box2[2])
  y2 = np.maximum(box1[3], box2[3])
  intersection = area([x, y, x1, y1])
  
  area1 = area(box1)
  area2 = area(box2)

  if intersection == area1:
    return a

  if intersection == area2:
    return b

  if area1 >= area2:
    if float(intersection) / float(area1) > 0.7:
      return a
  else:
    if float(intersection) / float(area2) > 0.7:
      return b

  return -1




