import cv2
import matplotlib.pyplot as plt
from recoplate.models import PlateRecognition
from config.config import MODEL_CONFIGURATION
from recoplate.utils import draw_first_plate

model_configuration = "motorcycle"
model_configuration = MODEL_CONFIGURATION[model_configuration]
model = PlateRecognition(
    model_configuration["detector_name"],
    model_configuration["ocr_method"]
    )

device = 0
cap = cv2.VideoCapture(device)

if not cap.isOpened():
    print(f"Cannot open {device}")
    exit()

print("Starting to read license plate")
while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frames. Exiting...")
        break

    cropped_plate, all_plate_text = model.predict(frame)

    # if cropped_plate:
    #     frame = draw_first_plate(frame, cropped_plate, all_plate_text)

    cv2.imshow("plate-detected", frame)

    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
