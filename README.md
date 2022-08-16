## facereglib
This is a very simple face recognition package which adapt from [deepface](https://github.com/serengil/deepface) (basically the model and its weights is the same as deepface). I just remove the face analysis part and narrow it for only the face recognition task. You might want to visit [deepface](https://github.com/serengil/deepface) to get a complete face recognition framework.

## Installation
To use facereglib, clone the facereglib repository to your working directory:
```bash
git clone https://github.com/thaitran24/facereglib.git
```
Make sure to install required packages in `requirements.txt` with:
```bash
pip install -r requirements.txt
```

### With Docker
`Update later`

## Features

### Models
Thís package contains some face recognition models: [`DeepID`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/deepid.py), [`DeepFace`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/deepface.py), [`OpenFace`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/openface.py), [`VGG-Face`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/vggface.py), [`Facenet`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/facenet.py), [`Facenet512`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/facenet512.py), [`ArcFace`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/arcface.py). This package contains 2 major objects: a `recognizer` which recognizes faces base on builded database features of method `buildDatabase()` and a `detector` which detects faces with Google's [`mediapipe`](https://github.com/thaitran24/facereglib/blob/master/facereg/detector).

### Recognition
```python
from facereglib.facereg import recognizer

models = [
  "vggface", 
  "facenet", 
  "facenet512", 
  "openface", 
  "deepface", 
  "deepid", 
  "arcface", 
]

# initialize face recognition model
model = reconizer.Recognizer(model_name=models[1])
```
The model first creates the `weights` directory in `facereglib` to store weights of the model:
```bash
project
├── facereglib
│   ├── facreg
│   │   ├── ...
│   ├── utils
│   │   ├── ...
│   ├── weights
│   │   ├── model_weights.h5
```
You can create the `weights` directory then download and save the weight from [GDrive](https://drive.google.com/drive/folders/1QK40h-D8DmREWdLPMbnTy5gCLsO3WE-1?usp=sharing). If not, the model will automatically download and save the weight for you.

The `recognizer` takes images as `numpy` array. We can verify whether 2 faces are same or not with `verify()`;
```python
# face verification
result = model.verify(img1, img2)
```

However, to perform face recognition, you need to build the database face features representation first. You should arrange the directory of the database as illustrated below:
```bash
project
├── database
│   ├── Person1
│   │   ├── person1.1.jpg
│   │   ├── person1.2.jpg
│   ├── Person2
│   │   ├── person2.1.jpg
```
Then the `buildDatabase()` method extract the features of each image and save the image path and its features in a representation file in `src_path` for recognition task. 

```python
database_folder = '/database/'
representation_folder = '/represent/'

# build database features
model.buildDatabase(db_path=database_folder, db_represent_path=representation_folder)

# face recognition: find the most similar face in database
df = model.recognize(img)   # return dataframe ['identity', 'distance']

# find identities: find and sort similarity in ascending order of 'distance'
df = model.find(img)
```

If you've already had a representation file and don't want to rebuild the database, you can initialize the model with representation file's directory:
```python
representation_folder = '/represent/'

model = reconizer.Recognizer(model_name=models[1], db_represent_path=representation_folder)
```
Then you can perform `find()` and `recognize()` method normally.