#!/usr/bin/env python3


import numpy as np
Yolo = __import__('1-yolo').Yolo

np.random.seed(1)
anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]])
yolo = Yolo('test.h5', 'test.txt', 0.6, 0.5, anchors)
output1 = np.random.randn(13, 13, 3, 85)
output2 = np.random.randn(26, 26, 3, 85)
output3 = np.random.randn(52, 52, 3, 85)
boxes, box_confidences, box_class_probs = yolo.process_outputs([output1, output2, output3], np.array([500, 700]))
print(str(boxes))
with open('1-test', 'w+') as f:
    f.write(str(box_confidences))
with open('2-test', 'w+') as f:
    f.write(str(box_class_probs))
