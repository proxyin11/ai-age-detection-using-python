import cv2
import imutils
import time
import numpy as np
import os
print(os.path.exists('data/age_deploy.prototxt'))
print(os.path.exists('data/age_net.caffemodel'))

cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width
cap.set(4, 640) #set height

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = [
    '(0–3)', '(4–7)', '(8–11)', '(12–15)',
    '(16–19)', '(20–24)', '(25–29)', '(30–34)',
    '(35–39)', '(40–44)', '(45–49)', '(50–54)',
    '(55–59)', '(60–64)', '(65–69)', '(70–100)'
]

gender_list = ['Male', 'Female']

def initialize_caffe_models():
	
	age_net = cv2.dnn.readNetFromCaffe(
    'data/age_deploy.prototxt',
    'data/age_net.caffemodel'
)

	return(age_net)

def read_from_camera(age_net):
	font = cv2.FONT_HERSHEY_SIMPLEX

	while True:

		ret, image = cap.read()
		
		

		face_cascade = cv2.CascadeClassifier("C:/Users/varsa/AppData/Roaming/Python/Python313/site-packages/cv2/data/haarcascade_frontalface_alt.xml")

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)		

		if(len(faces)>0):
			print("Found {} faces".format(str(len(faces))))
		else:cv2.destroyAllWindows()
		

		for (x, y, w, h )in faces:	
			cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

			# Get Face 
			face_img = image[y:y+h, x:x+w].copy()
			blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

			

			#Predict Age
			age_net.setInput(blob)
			age_preds = age_net.forward()
			age = age_list[age_preds[0].argmax(0)]
			print("Age Range: " + age)

			overlay_text = "%s" % age
			cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


		cv2.imshow('frame', image)

		if cv2.waitKey(0) & 0xFF == ord('q'):
			break
		

if __name__ == "__main__":
	age_net = initialize_caffe_models()

	read_from_camera(age_net)