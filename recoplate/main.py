import cv2
import click

from models import PlateRecognition
from config.config import MODEL_CONFIGURATION

@click.group()
def cli():
    pass

@cli.command()  
@click.argument("device", default=0)
@click.argument("model_configuration", default="motorcycle")
def webcam(device, model_configuration):

    model_configuration = MODEL_CONFIGURATION[model_configuration]
    model = PlateRecognition(
        model_configuration["detector_name"],
        model_configuration["ocr_method"]
        )

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"Cannot open {device=}")
        exit()
    
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frames. Exiting...")
            break

        cropped_plate, all_plate_text = model.predict(frame)

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
            cv2.imshow("frame", plate_detected)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    ...

if __name__ == "__main__":
    cli()
