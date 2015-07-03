# import the necessary packages
import numpy as np

def area(box):
  return (abs(box[2] - box[0])) * (abs(box[3] - box[1]))

def overlaps(a, b, thresh=0.5):
  print "checking overlap "
  print a, b
  x1 = np.maximum(a[0], b[0])
  x2 = np.minimum(a[2], b[2])
  y1 = np.maximum(a[1], b[1])
  y2 = np.minimum(a[3], b[3])
  intersect = float(area([x1, y1, x2, y2]))
  return intersect / np.minimum(area(a), area(b)) >= thresh

def is_inside(rec1, rec2):
  def inside(a,b):
    if (a[0] >= b[0]) and (a[2] <= b[0]):
      return (a[1] >= b[1]) and (a[3] <= b[3])
    else:
      return False

  return (inside(rec1, rec2) or inside(rec2, rec1))

def non_max_suppression(boxes, overlap_thresh = 0.5):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return []

  scores = boxes[:,4]
  score_idx = np.argsort(scores)

  while len(score_idx) > 0:
    box = scores[score_idx[0]]
    print "checking box"
    for s in score_idx:
      to_delete = []
      if s == 0:
        continue
      try:
        if (overlaps(boxes[s], boxes[box], overlap_thresh)):
          to_delete.append(box)
          score_idx = np.delete(score_idx, [s], 0)
      except:
        pass
    boxes = np.delete(boxes, to_delete, 0)
    score_idx = np.delete(score_idx, 0, 0)

  return boxes
