import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image
from facereglib.utils import distance

def detect(img):
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_width = img.shape[1]
        img_height = img.shape[0]
        
        if not results.detections:
            return [], (0, img_width, 0, img_height)

        faces = []
        regions = []
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = max(int(bbox.xmin * img_width), 0)
            y = max(int(bbox.ymin * img_height), 0)
            w = int(bbox.width * img_width)
            h = int(bbox.height * img_height)

            # Get keypoints
            keypoints = detection.location_data.relative_keypoints
            for keypoint in keypoints:
                keypoint.x = int(keypoint.x * img_width) - x
                keypoint.y = int(keypoint.y * img_height) - y

            face = img[y : y + h, x : x + w]
            face = align(face, keypoints)
            faces.append(face)
            regions.append((x, x + w, y, y + h))

    return faces, regions


def align(img, keypoints):
    """
    Perform face alignment (2D) with keypoints
    Parameters:
        img: face deteced img
        keypoints: relative face detected keypoints for alignment
    Return:
        Aligned face img
    """
    left_eye = (keypoints[1].x, keypoints[1].y)
    right_eye = (keypoints[0].x, keypoints[0].y)
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    if left_eye_y > right_eye_y:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 # Rotate inverse direction of clock
    else:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 # Rotate same direction to clock
    
    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    if b != 0 and c != 0: # This multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a) # angle in radian
        angle = (angle * 180) / math.pi # radian to degree
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    
    return img 