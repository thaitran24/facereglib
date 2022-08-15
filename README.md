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
Update later

## Features

### Models
Thís package contains some face recognition models: [`DeepID`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/deepid.py), [`DeepFace`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/deepface.py), [`OpenFace`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/openface.py), [`VGG-Face`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/vggface.py), [`Facenet`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/facenet.py), [`Facenet512`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/facenet512.py), [`ArcFace`](https://github.com/thaitran24/facereglib/blob/master/facereg/models/arcface.py). This package contains 2 major objects: a `recognizer` which recognizes faces base on builded database features of method `buildDatabase()` and a `detector` which detects faces with Google's [`mediapipe`](https://github.com/thaitran24/facereglib/blob/master/facereg/detector).

### Recognition
The `recognizer` object can verify 2 `numpy` array face images:
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

# face verification
result = model.verify(img1, img2)
```

However, to perform face recognition, we need to build the database face features representation first.

```python
database_folder = '/database/'
representation_folder = '/represent/'

# build database features
model.buildDatabase(db_path=database_folder, src_path=representation_folder)

# face recognition
df = model.recognize(img)
```