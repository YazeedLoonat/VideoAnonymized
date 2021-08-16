# USAGE
# python blur_face_video.py --video examples/test2.mp4

from pyimagesearch.face_blurring import anonymize_face_pixelate
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video")
ap.add_argument("-f", "--face", default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-b", "--blocks", type=int, default=20,
	help="# of blocks for the pixelated blurring method")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] Start")
cap = cv2.VideoCapture(args["video"])
success, img = cap.read()

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

wri = cv2.VideoWriter(
	'output/example.avi',
	cv2.VideoWriter_fourcc(*'MJPG'),
	int(cap.get(cv2.CAP_PROP_FPS)),
	(width, height)
)

face_cascade = cv2.CascadeClassifier('face_detection.xml')

# loop over the frames from the video stream
while success:
	frame = cv2.resize(img, (width, height))
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	for (x, y, w, h) in faces:
		face = frame[y:(y+h), x:(x+w)]
		face = anonymize_face_pixelate(face, blocks=args["blocks"])
		frame[y:(y+h), x:(x+w)] = face
	wri.write(frame)
	success, img = cap.read()
cap.release()
wri.release()
cv2.destroyAllWindows()
print("[INFO] End")