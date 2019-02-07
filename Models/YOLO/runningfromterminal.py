import os
import cv2
from matplotlib import pyplot as plt

def yolo3_detection(image):
	res = os.system('./darknet detect cfg/yolov3.cfg yolov3.weights Random/{0}'.format(image))
	img = cv2.imread('predictions.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	plt.imshow(img)
	plt.show()
	return

if __name__ == '__main__':

	print('--------')
	print('YOLO Version3 for Detection of Humans in an image')
	print('--------')
	while(True):
		inp = input('Enter Image name(with extension): Type DONE to Exit\n')
		if(inp != 'DONE' and inp != 'done'):
			yolo3_detection(inp)
		else:
			break