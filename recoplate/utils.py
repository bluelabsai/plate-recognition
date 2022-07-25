import re
from typing import Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf

from config.config import TRANSFORM_LASTCHAR

def load_tf_model(model_path:Path) -> Tuple:
    labelmap_file = Path(model_path, "label_map.pbtxt")
    save_model_dir = Path(model_path, "saved_model")

    detect_fn = tf.saved_model.load(str(save_model_dir))

    return detect_fn


def ocr_to_motorplate(raw_plate):
  plate = re.sub('[^a-zA-Z0-9]', '', raw_plate).upper()

  # check hard rules
  if len(plate) != 6:
    return
  
  last_char = plate[-1]

  last_char_validated = TRANSFORM_LASTCHAR.get(last_char, last_char)
  plate = plate[:-1] + last_char_validated

  return plate


def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate
