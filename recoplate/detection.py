from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf

from config.config import MODELS_DIR
from recoplate.utils import load_tf_model

ALLOW_DETECTOR_MODELS = ["mobilenet"]


class PlateDetection:

    def __init__(self, model_name: str):

        if model_name not in ALLOW_DETECTOR_MODELS:
            raise ValueError(
                f"model {model_name} is not implemented, try someone: {ALLOW_DETECTOR_MODELS}"
                )
        
        self.model_dir = Path(MODELS_DIR / model_name)

        self.load_model()

    def load_model(self):
        self.labels, self.detector_model = load_tf_model(self.model_dir)


    def preprocess(self, frame: np.ndarray) -> tf.Tensor:

        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        return input_tensor


    def filter_detection(self, frame: np.ndarray, detections: Dict) -> List:

        image = np.array(frame)
        scores = list(
            filter(lambda x: x>self.object_detection_trh , detections["detection_scores"])
            )
        boxes = detections["detection_boxes"][:len(scores)]
        # classes = detections["detection_classes"][:len(scores)]

        width = image.shape[1]
        height = image.shape[0]

        cropped_plate = []
        for box in boxes:
            roi = box*[height, width, height, width]
            cropped_plate.append(image[int(roi[0]): int(roi[2]), int(roi[1]):int(roi[3])])

        return cropped_plate


    def _predict(self, frame: np.ndarray) -> np.ndarray:

        input_tensor = self.preprocess(frame)
        detections = self.detector_model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        cropped_plate = self.filter_detection(frame, detections)

        return (cropped_plate, detections)


    def predict(
        self, frame: np.ndarray, object_detection_trh: float=0.8, ocr_threshold: float=0.5
        ) -> np.ndarray:
        self.object_detection_trh = object_detection_trh
        cropped_plate, detections = self._predict(frame)

        return (cropped_plate, detections)
