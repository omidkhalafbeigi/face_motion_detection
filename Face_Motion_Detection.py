import cv2 as cv
import numpy as np


def get_detected_body(classifier, frame):
    img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (5, 5), cv.BORDER_DEFAULT)

    detected_body = classifier.detectMultiScale(img_blur)

    if len(detected_body) > 0:
        detected_body = detected_body[0]
        x, y, w, h = detected_body
        body = img_blur[y:y + h, x:x + w]
        if body.shape[1] > 0 and body.shape[0] > 0:
            body_rgb = cv.cvtColor(body, cv.COLOR_BGR2RGB)
            return body_rgb


cascade_classifier = cv.CascadeClassifier(cv.data.haarcascades + '/haarcascade_frontalface_default.xml')
video = cv.VideoCapture(0)
frame_counter = 0
shoot_threshold = 20
frame_threshold = 2
frame_start = 10
motion_detected = False
diff = first_frame_contours_mean = next_frame_contours_mean = -1

while True:
    _, frame_read = video.read()

    diff = 0
    if frame_counter <= frame_start:
        frame_counter += 1
        first_frame_body = get_detected_body(cascade_classifier, frame_read)
        if first_frame_body is None:
            frame_counter = 0
        else:
            x_first_center = first_frame_body.shape[0] / 2
            y_first_center = first_frame_body.shape[1] / 2
    elif frame_counter > frame_start:
        next_frame_body = get_detected_body(cascade_classifier, frame_read)
        if next_frame_body is not None:
            x_new_size = min(first_frame_body.shape[0], next_frame_body.shape[0])
            y_new_size = min(first_frame_body.shape[1], next_frame_body.shape[1])
            diff = cv.absdiff(cv.resize(first_frame_body.copy(), (x_new_size, y_new_size)),
                              cv.resize(next_frame_body.copy(), (x_new_size, y_new_size)))
            diff = np.mean(diff)
            print(diff)
            print('---------------')
            cv.imshow('Frame', next_frame_body)
    if diff > shoot_threshold:
        motion_detected = True
        break

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

if motion_detected:
    print('Shoot!!!')
else:
    print('Not moved')

video.release()
cv.destroyWindow('Frame')
