from time import sleep

import cv2
import click

from config.config import MODEL_CONFIGURATION
from recoplate.models import PlateRecognition


@click.group()
def cli():
    pass

@cli.command()  
@click.argument("model_configuration", default="motorcycle")
@click.argument("device", default=0)
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
    
    print("Starting to read license plates")
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frames. Exiting...")
            break

        cropped_plate, all_plate_text = model.predict(frame)

        for plate_detected, text in zip(cropped_plate, all_plate_text):
            print(text)
            #sleep(1)
            # cv2.putText(
            #     plate_detected,
            #     text,
            #     (100,100),
            #     fontFace=0,
            #     fontScale=1,
            #     color=(0,255,0),
            #     thickness=2,
            #     lineType=cv2.LINE_AA
            #     )
            # plt.imshow(plate_detected)
            # plt.show()
            # if cv2.waitKey(1) == ord("q"):
            #     break

    cap.release()
    cv2.destroyAllWindows()

    ...

if __name__ == "__main__":
    cli()
