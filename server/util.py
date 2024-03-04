import cv2
import numpy as np
import base64
from wavelet import w2d
import json
import pickle

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def load_saved_artifacts():
    print("Loading saved artifacts")
    global __class_name_to_number
    global __class_number_to_name

    with open('./artifacts/class_dict.json', 'r') as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/model.pkl', 'rb') as f:
            __model = pickle.load(f)
    print("Successfully loaded saved artifacts")


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def crop_face_from_img(image_base64_str, image_path):
    face_cascade = cv2.CascadeClassifier('../model/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('../model/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_str)

    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.3, minNeighbors=6, minSize=(40, 40))

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_grey = grey_img[y:y+h, x:x+w]
        roi_colour = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey)
        if len(eyes) >= 2:
            cropped_faces.append(roi_colour)
    return cropped_faces


def classify_image(image_base64_str, file_path=None):
    preds = []
    imgs = crop_face_from_img(image_base64_str, file_path)
    for img in imgs:
        scaled_img = cv2.resize(img, (32, 32))
        wavelet_img = w2d(img, 'db1', 5)
        scaled_wavelet_img = cv2.resize(wavelet_img, (32, 32))
        stacked_img = np.vstack((scaled_img.reshape(-1, 1), scaled_wavelet_img.reshape(-1, 1)))
        final_img = stacked_img.reshape(1, -1).astype(float)
        preds.append(class_number_to_name(__model.predict(final_img)[0]))
    return preds


# def get_b64_test_image_for_lionelmessi():
#     with open("b64.txt") as f:
#         return f.read()


# if __name__ == '__main__':
    # load_saved_artifacts()
    # print(classify_image(get_b64_test_image_for_lionelmessi()))
