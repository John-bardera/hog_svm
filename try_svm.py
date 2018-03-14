# coding:utf-8

import os
import time
import numpy as np
import cv2
from selectivesearch import selective_search as ss

start_time = time.time()

win_size = (60, 60)
block_size = (16, 16)
block_stride = (4, 4)
cell_size = (4, 4)
bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, bins)

target_path = "/mnt/c/Users/user/Pictures/dataset_favorite/"
svm = cv2.ml.SVM_load("train.xml")

target = os.listdir(target_path)[0]
print(target)

img_bgr = cv2.imread(target_path + target)
img = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2GRAY)
#resize_size = (int(img_bgr.shape[0]), int(img_bgr.shape[1]))
#img = cv2.resize(img, resize_size)
#img_bgr_resize = cv2.resize(img_bgr.copy(), resize_size)


img_load_time = time.time()
print(f"img_load:{img_load_time - start_time}")

img_lbl, regions = ss(img_bgr, scale=500, sigma=0.9, min_size=1000)

ss_time = time.time()
print(f"ss:{ss_time - img_load_time}")

hog_list = []
hog_list_rect = []
almosit_regions = len(regions)

for index, i in enumerate(regions):
	x, y, w, h = i["rect"]
	if h < win_size[1] or w < win_size[0]:
		continue
	print(f"hog: {index + 1}/{almosit_regions}")
	img_temp = cv2.resize(img[y:y+h, x:x+w].copy(), win_size)
	hog_list.append(hog.compute(img_temp))
	hog_list_rect.append((x, y, w, h))

result = svm.predict(np.array(hog_list))[1].ravel()

hog_and_svm_time = time.time()
print(f"hog_and_svm:{hog_and_svm_time - ss_time}")

print(result)

recog_face_rect = [hog_list_rect[index] for index, i in enumerate(result) if i == 0.]

color = (0, 0, 255)

if recog_face_rect:
	for i in recog_face_rect:
		x, y, w, h = i
		cv2.rectangle(img_bgr, (x,y), (x+w, y+h), color, thickness=8)
	cv2.imwrite(f"./ppp.jpg", img_bgr)

end_time = time.time()
print(f"end:{end_time - hog_and_svm_time}")

print(f"total:{end_time - start_time}")
