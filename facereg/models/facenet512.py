from facenet import InceptionResNetV2
import os

def loadModel():
    facenet_model = InceptionResNetV2(dimension=512)

    file_name = os.getcwd() + '/facereglib/weights/facenet512_weights.h5'
    facenet_model.load_weights(file_name)

    return facenet_model