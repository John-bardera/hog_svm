# coding:utf-8

import os
import cv2
from PIL import Image

img_dir = "/mnt/c/Users/user/Pictures/dataset_favorite/"

file_list = os.listdir(img_dir)
#cascade_file = "../develop3_5_2/lib/python3.5/site-packages/cv2/data/haarcascade_eye.xml"#frontalface_alt.xml"
cascade_file = "./lbpcascade_animeface.xml"

datas = []

for file in file_list:
	split = file.split(".")
	if not split[-1] in ["jpg", "jpeg", "JPG", "png", "PNG", "JPEG"]:
		continue
	img = cv2.imread(img_dir + file)
	img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	cascade = cv2.CascadeClassifier(cascade_file)
	face_list = cascade.detectMultiScale(
		img_gs,
		scaleFactor=1.1,
		minNeighbors=1,
		minSize=(30, 30))

	if len(face_list) > 0:
		#color = (0, 0, 255)
		for index, face in enumerate(face_list):
			x,y,w,h = face
			file_name = f"./dataset/{index}{file}"
			Image.open(img_dir + file).crop((x, y, x + w, y + h)).convert("RGB").save(file_name)
			datas.append(file_name)
			#cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness=8)
			#cv2.imwrite(f"./face/{file}", img)
	else:
		cv2.imwrite(f"./no_face/{file}", img)

f = open("./dataset_data.py", "w")
f.write("pos_file_name = [")
for i in datas:
	f.write(f"{i},\n\t")
f.write("]")
