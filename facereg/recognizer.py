from facereglib.utils import preprocess, distance
from facereglib.facereg import detector
import numpy as np
import os
import pickle
from tqdm import tqdm
from PIL import Image
import pandas as pd
from pathlib import Path

from facereglib.facereg.models import vggface
from facereglib.facereg.models import deepface
from facereglib.facereg.models import deepid
from facereglib.facereg.models import arcface
from facereglib.facereg.models import openface
from facereglib.facereg.models import facenet

class Recognizer():
    def __init__(self, model_name, db_represent_src=None) -> None:
        models = {
            'vggface': vggface.loadModel,
            'deepface': deepface.loadModel,
            'deepid': deepid.loadModel,
            'arcface': arcface.loadModel,
            'facenet': facenet.loadModel,
            'openface': openface.loadModel
        }
        base_model = models.get(model_name)
        if not base_model:
            raise ValueError('Invalid model_name passed - {}'.format(model_name))
        
        self.model = base_model()
        self.model_name = model_name
        self.db_represent_src = db_represent_src
        self.is_db_build = True if db_represent_src != None else False

    
    def find(self, img, distance_metric='cosine', threshold=0.3, top_rows=5):
        if not self.is_db_build:
            raise FileNotFoundError("There is no database representation file. Please build database first by calling: buildDatabase()") 

        representation_file = open(self.db_represent_src + 'representation.pkl', 'rb')
        representations = pickle.load(representation_file)
        df = pd.DataFrame(representations, columns=['identity', 'representation'])
        face = self.represent(img)
        distances = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            src_rep = row['representation']
            if distance_metric == 'euclidean':
                dist = distance.findEuclideanDistance(face, src_rep)
            elif distance_metric == 'cosine':
                dist = distance.findCosineDistance(face, src_rep)
            else:
                raise ValueError("Invalid distance_metric passed - ", distance_metric)
            distances.append(dist)

        threshold = preprocess.findThreshold(self.model_name, distance_metric)
        df['distance'] = distances
        df = df.drop(columns = ['representation'])
        df = df[df['distance'] <= threshold]
        df = df.sort_values(by = ['distance'], ascending=True).reset_index(drop=True)
        n_rows = min(len(df.index), top_rows)
        return df.head(n_rows)


    def represent(self, img):
        self.input_size = self.model.layers[0].input_shape[0][1:3]
        faces, regions = detector.detect(img)
        if len(faces) == 0:
            face = preprocess.resize(img, self.input_size)
        else:
            face = preprocess.resize(faces[0], self.input_size)
        # face = preprocess.normalize(face, normalization=self.model_name)
        return self.model.predict(face)[0].tolist()
    

    def verify(self, img1, img2, threshold=0.2, distance_metric='cosine'):
        face1 = self.represent(img1)
        face2 = self.represent(img2)
            
        threshold = preprocess.findThreshold(self.model_name, distance_metric)
        if distance_metric == 'cosine':
            dist = distance.findCosineDistance(face1, face2)
        elif distance_metric == 'euclidean':
            dist = distance.findEuclideanDistance(face1, face2)
        else:
            raise ValueError("Invalid distance_metric passed - ", distance_metric)

        dist = np.float64(dist)
        return True if dist <= threshold else False
    

    def buildDatabase(self, db_path, src_path):
        if not os.path.isdir(db_path):
            print("Database path db_path - ", db_path, " not exist")
            return

        file_name = 'representation.pkl'
        if os.path.exists(src_path + '/' + file_name):
            f = open(src_path + '/' + file_name, 'rb')
            representations = pickle.load(f)
        
        employees = []
        for rt, dr, fs in os.walk(db_path):
            for file in fs:
                if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                    exact_path = rt + '/' + file
                    employees.append(exact_path)

        if len(employees) == 0:
            raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this path.")
        
        representations = []
        pbar = tqdm(range(0, len(employees)), desc='Building representation')
        
        for index in pbar:
            employee = employees[index]
            img = Image.open(employee)
            img = np.asarray(img)
            representations.append([employee, self.represent(img)])
        
        Path(src_path).mkdir(parents=True, exist_ok=True)
        self.db_represent_src = src_path + '/' + file_name
        representation_file = open(self.db_represent_src, 'wb')
        pickle.dump(representations, representation_file)
        self.is_db_build = True
        representation_file.close()
    

    def recognize(self, img, threshold=0.3, distance_metric='cosine'):
        if not self.is_db_build:
            raise FileNotFoundError("There is no database representation file. Please build database first by calling: buildDatabase()") 

        df = self.find(img, distance_metric, threshold, top_rows=5)
        return df