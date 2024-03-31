import numpy as np
import cv2
import os
import time
import serial

# Function to calculate distance between vectors

port = serial.Serial('/dev/cu.usbserial-120', 9600)
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

# Function to implement KNN algorithm
def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Path to the face dataset
dataset_path = "./face_dataset/"

# Load face data and labels
face_data = []
labels = []
class_id = 0
names = {}

# Dataset preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))

trainset = np.concatenate((face_dataset, face_labels), axis=1)

# Define the threshold value
threshold = 100

# Main loop for face detection and recognition
while True:
    ret, frame = cap.read()
    if ret == False:
        continue
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # Get the face ROI
        offset = 5 

        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Predict the label of the detected face
        out = knn(trainset, face_section.flatten())
        # Check if the detected face matches any of the trained faces
        matched = False
        for label in trainset[:, -1]:
            if int(out) == int(label):
                matched = True
                break

        # Set name to "Unrecognized" if the detected face doesn't match any trained faces
        if not matched:
            name = "Unrecognized"
            port.write(str.encode('0'))
        else:
            name = names[int(out)]
            if (name == "devashish"):
                port.write(str.encode('1'))
                print("data sent")
            elif (name == "yash"):
                port.write(str.encode('2'))
                print("data sent")
            elif (name == "iftikhar"):
                port.write(str.encode('3'))
                print("data sent")

        # Print the name associated with the predicted label
        print("Face detected of:", name)

        # Draw rectangle around the face and display name
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)


    # Display the frame
    cv2.imshow("Faces", frame)

    # Check for exit command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()