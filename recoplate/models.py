import numpy as np
from typing import Dict, List, Tuple

from detection import PlateDetection
from recognition import OCRMotorModel

class PlateRecognition:
    def __init__(self, detector_name, ocr_method) -> None:
        self.detection = PlateDetection(detector_name)
        self.recognition = OCRMotorModel(ocr_method)
        
    def _predict(self, frame: np.ndarray) -> Dict:

        predict_data = {}

        cropped_plate, detections = self.detection.predict(frame, self.object_detection_trh)

        all_plate_text = []
        scores = []
        for crop_plate in cropped_plate:
            plate_text, score = self.recognition.predict(crop_plate, self.ocr_threshold)
            all_plate_text.append(plate_text)
            scores.append(score)
        
        predict_data["detection"] = (cropped_plate, detections)
        predict_data["recognition"] = (all_plate_text, scores)

        return (cropped_plate, all_plate_text, predict_data)

    def predict(
        self, frame: np.ndarray, object_detection_trh: float=0.8,  ocr_threshold: float=0.5
        ) -> List:

        self.object_detection_trh = object_detection_trh
        self.ocr_threshold = ocr_threshold

        cropped_plate, all_plate_text, _ = self._predict(frame)

        return [cropped_plate, all_plate_text]



