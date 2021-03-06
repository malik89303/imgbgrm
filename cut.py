from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from array import array
import math as math
import os
import sys
from io import BytesIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import datetime
import shutil

def viewImage(image, name_of_window):
	cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
	cv2.imshow(name_of_window, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def rotateImage(image, angle):
	height, width = image.shape[:2]
	image_center = (width / 2,height / 2)
	rotation_img = cv2.getRotationMatrix2D(image_center, angle, 1.)
	abs_cos = abs(rotation_img[0, 0])
	abs_sin = abs(rotation_img[0, 1])
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)
	rotation_img[0, 2] += bound_w / 2 - image_center[0]
	rotation_img[1, 2] += bound_h / 2 - image_center[1]
	rotated_img = cv2.warpAffine(image, rotation_img, (bound_w, bound_h))
	return rotated_img


def face_cutting(filename):
	image_path = filename
	image = cv2.imread(image_path)
	height, width, channels = image.shape
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	counterrotations = 0
	countereyes = 0
	while (counterrotations < 4) and (countereyes != 2):
		path = str('haarcascade_eye_tree_eyeglasses.xml')
		eyes_cascade = cv2.CascadeClassifier(path)
		eyes = eyes_cascade.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=( int(image.shape[0]*0.01), int(image.shape[0]*0.01))
		)
		countereyes = len(eyes)
		if countereyes != 2:
			image = rotateImage(image, 90)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			counterrotations += 1

	if countereyes != 2:
		sys.exit("Попробуйте использовать другое изображение.")

	if eyes[0][0] + eyes[0][2] // 2 > eyes[1][0] + eyes[1][2] // 2:
		eyes[0][0], eyes[1][0] = eyes[1][0], eyes[0][0]
		eyes[0][1], eyes[1][1] = eyes[1][1], eyes[0][1]
		eyes[0][2], eyes[1][2] = eyes[1][2], eyes[0][2]
		eyes[0][3], eyes[1][3] = eyes[1][3], eyes[0][3]
	rangle = math.asin(math.fabs((eyes[0][1] + eyes[0][3] // 2) - (eyes[1][1] + eyes[1][3] // 2)) / (math.sqrt(
		((eyes[1][0] + eyes[1][2] // 2) - (eyes[0][0] + eyes[0][2] // 2)) ** 2 + (
				(eyes[0][1] + eyes[0][3] // 2) - (eyes[1][1] + eyes[1][3] // 2)) ** 2)))
	if (eyes[0][1] + eyes[0][3] // 2 - eyes[1][1] + eyes[1][3] // 2) > 0.1 * height:
		image = rotateImage(image, 360 - rangle * 180 / math.pi)
		gray = rotateImage(gray, 360 - rangle * 180 / math.pi)
	elif (-eyes[0][1] + eyes[0][3] // 2 + eyes[1][1] + eyes[1][3] // 2) > 0.1 * height:
		image = rotateImage(image, rangle * 180 / math.pi)
		gray = rotateImage(gray, rangle * 180 / math.pi)
	path1 = str('haarcascade_frontalface_alt.xml')
	facecascade = cv2.CascadeClassifier(path1)
	faces = facecascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	x, y, w, h = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(image) 
	img = img.crop((x - 0.27 * w, y - 0.5 * h, x + 1.27 * w, y + 1.5 * h))	
	img = img.resize((399, 531)) 

	image = np.array(img)
	image = image[:, :, ::-1].copy()
	Path("aftercutting").mkdir(parents=True, exist_ok=True)
	cv2.imwrite("aftercutting/" + "temp.jpg", image)


class DeepLabModel(object):
	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, tarball_path):
		self.graph = tf.Graph()

		graph_def = None
		graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

		if graph_def is None:
			raise RuntimeError('Ошибка обработки.')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.Session(graph=self.graph)

	def run(self, image):
		startts = datetime.datetime.now()
		width, height = image.size
		resized = 1.0 * self.INPUT_SIZE / max(width, height)
		targ = (int(resized * width), int(resized * height))
		resimg = image.convert('RGB').resize(targ, Image.ANTIALIAS)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resimg)]})
		segmap = batch_seg_map[0]

		endts = datetime.datetime.now()

		timediff = endts - startts
		print("Затрачено времени на обратботку: " + str(timediff))
		return resimg, segmap

def drawSegment(baseImg, matImg,outputfp):
	width, height = baseImg.size
	nobgimg = np.zeros([height, width, 4], dtype=np.uint8)
	for x in range(width):
		for y in range(height):
			color = matImg[y,x]
			(r,g,b) = baseImg.getpixel((x,y))
			if color == 0:
				nobgimg[y,x] = [255,255,255,0]
			else :
				nobgimg[y,x] = [r,g,b,255]
	img = Image.fromarray(nobgimg)
	img.convert('RGB').save(outputfp,"jpeg")
	print("Файл сохранен в директории со скриптом: "+outputfp)
	shutil.rmtree("aftercutting")



def start_vis(filepath,outputfp):
  try:
  	print("Пытаюсь открыть : " + sys.argv[1])
  	jpeg_str = open(filepath, "rb").read()
  	orignal_im = Image.open(BytesIO(jpeg_str))
  except IOError:
    print('Невозможно извлечь изображение. проверьте файл: ' + filepath)
    return
  print('Запуск deeplab на %s...' % filepath)
  resized_im, seg_map = MODEL.run(orignal_im)
  drawSegment(resized_im, seg_map,outputfp)
  
  
modelType = "model"
MODEL = DeepLabModel(modelType)

inputfp = sys.argv[1]
outputfp = "done"+Path(inputfp).stem +".jpg"


if inputfp is None :
	print("Неверный ввод. Проверьте расположение файла")
	exit()
if os.path.isfile(inputfp):
	extensions = ['jpg', 'jpeg']
	file = inputfp.split('.')
	if len(file) >= 2:
		fileExtension = file[-1].lower()
		if fileExtension in extensions:
			print("Файл успешно найден. начинается обработка")
			face_cutting(inputfp)
			print("Фотграфия успешно повернута и обрезана. Запуск удаления фона")
			start_vis("aftercutting/temp.jpg",outputfp)
		else:
			sys.exit("Разерешены только файлы формата jpeg и jpg")
	else:
		sys.exit("У файла отстутсвует расширение. попробуйте еще раз")
else:
	print("Файл не найден. попробуйте еще раз")



