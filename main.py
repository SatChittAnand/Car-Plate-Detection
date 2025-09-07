from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
import util
# from sort.py.sort import *
from sort import *
from util import get_car, read_license_plate, write_csv


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./license_plate_detector.pt')
# license_plate_detector = YOLO('/kaggle/input/licenseplatedetection/LicensePlateDetection.pt')

# load video
cap = cv2.VideoCapture('sample.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        # track_ids = mot_tracker.update(np.asarray(detections_))
        if len(detections_) > 0:
            dets_np = np.asarray(detections_)
        else:
            dets_np = np.empty((0, 5))

        track_ids = mot_tracker.update(dets_np)

        # detect license plates
        # license_plates = license_plate_detector(frame)[0]
        # for license_plate in license_plates.boxes.data.tolist():
        #     x1, y1, x2, y2, score, class_id = license_plate

        #     # assign license plate to car
        #     xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        #     if car_id != -1:

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Add this print statement:
            print(f"Frame: {frame_nmr}, License Plate Bbox: {[x1, y1, x2, y2]}, Car ID: {car_id}")

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './test.csv')


# from ultralytics import YOLO
# import cv2

# model = YOLO("best.pt")
# img = cv2.imread("car-plate.jpg")  # an image with a plate
# results = model(img)
# results[0].show()          # display detections in a window
# results[0].save("out.jpg") # save annotated image
