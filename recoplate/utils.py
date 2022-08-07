import re
import zipfile
from typing import Tuple
from pathlib import Path

import cv2
import gdown
import numpy as np
import tensorflow as tf

from config.config import TRANSFORM_CHAR, URL_MODELS, MODELS_DIR

def load_tf_model(model_path:Path) -> Tuple:
    labelmap_file = Path(model_path, "label_map.pbtxt")
    save_model_dir = Path(model_path, "saved_model")

    detect_fn = tf.saved_model.load(str(save_model_dir))

    return detect_fn


def replace_char(chars, transform_dic):
  for s_char, r_char in transform_dic.items():
    chars = chars.replace(s_char, r_char)
  return chars


def ocr_to_motorplate(raw_plate):
  plate = re.sub('[^a-zA-Z0-9]', '', raw_plate).upper()

  transform_chars_motor = TRANSFORM_CHAR["motorcycle"]

  # check hard rules
  if len(plate) != 6:
    return
  
  initian_chars_r = replace_char(plate[:3], transform_chars_motor["initial_char"])
  middle_chars_r = replace_char(plate[3:5], transform_chars_motor["middle_char"])
  last_char_r = replace_char(plate[-1], transform_chars_motor["last_char"])
  
  plate = initian_chars_r + middle_chars_r + last_char_r

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


def download_models(model_name: str):
  print("download fold from Google Drive ...")
  file_id = URL_MODELS[model_name]
  destination = Path(MODELS_DIR, model_name)

  gdown.download_folder(
    id=file_id,
     output=str(destination),
     use_cookies=False
)

def resize_img(img, width, height):
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def draw_first_plate(img, cropped_plate, all_plate_text, ):
    x = int(img.shape[0] * 0.6)
    y = int(img.shape[1] * 0.1)

    # size for show plate
    width = int(img.shape[0] * 0.35)
    height = int(img.shape[1] * 0.15)

    plate = cropped_plate[0]
    plate_text = all_plate_text[0]

    plate = resize_img(plate, width, height)
    cv2.putText(img, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (255, 10, 58), 3)
    img[y+10:y+10+height, x:x+width] = plate

    return img
