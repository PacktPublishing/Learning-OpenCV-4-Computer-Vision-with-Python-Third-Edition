def sliding_window(image, step, window_size):
  for y in xrange(0, image.shape[0], step):
    for x in xrange(0, image.shape[1], step):
      yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
