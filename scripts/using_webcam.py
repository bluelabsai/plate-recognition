import cv2
import matplotlib.pyplot as plt
from recoplate.models import PlateRecognition
from config.config import MODEL_CONFIGURATION

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

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frames. Exiting...")
        break

    cropped_plate, all_plate_text = model.predict(frame)

    print(all_plate_text)

    for plate_detected, text in zip(cropped_plate, all_plate_text):
        cv2.putText(
            plate_detected,
            text,
            (100,100),
            fontFace=0,
            fontScale=1,
            color=(0,255,0),
            thickness=2,
            lineType=cv2.LINE_AA
            )

        plt.imshow(plate_detected)
        plt.show()
        # cv2.imshow("plate-detected", plate_detected)

        # if cv2.waitKey(1) == ord("q"):
        #     break

cap.release()
cv2.destroyAllWindows()
