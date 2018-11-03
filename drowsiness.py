from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import serial
from goprocam import GoProCamera
from goprocam import constants


def eye_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    er = (a + b) / (2.0 * c)
    return er

EYE_AR_CONSEC_FRAMES = 14

COUNTER = 0
ALARM_ON = False

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=1).start()
time.sleep(1.0)
user_th = True
th_list = []

arduino = serial.Serial('/dev/cu.usbmodem1411', 9600)


def alert():
    arduino.write(str.encode('1'))

while user_th:
    img = vs.read()
    img = imutils.resize(img, width=1000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = detector(gray, 0)
    for rs in r:
        cv2.putText(img, "Please look into the camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        shape = predictor(gray, rs)
        shape = face_utils.shape_to_np(shape)
        rightEye = shape[r_start:r_end]
        leftEye = shape[l_start:l_end]
        right_eye_ratio = eye_ratio(rightEye)
        left_eye_ratio = eye_ratio(leftEye)
        ear = (left_eye_ratio + right_eye_ratio) / 2.0
        if len(th_list) > 50:
            user_th = False
        else:
            th_list.append(ear)
        rightEyeHull = cv2.convexHull(rightEye)
        leftEyeHull = cv2.convexHull(leftEye)
        cv2.drawContours(img, [leftEyeHull], -1, (212, 66, 244), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (212, 66, 244), 1)
        # cv2.putText(img, "{:.2f}".format(ear), (300, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("IMG", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

aver_EAR = sum(th_list) / len(th_list)
EYE_AR_THRESH = aver_EAR

while True:
    img = vs.read()
    img = imutils.resize(img, width=1000)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = detector(gray, 0)
    for rs in r:
        cv2.putText(img, "User's threshold is: " + str("{:.2f}".format(EYE_AR_THRESH)), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        shape = predictor(gray, rs)
        shape = face_utils.shape_to_np(shape)
        rightEye = shape[r_start:r_end]
        leftEye = shape[l_start:l_end]
        right_eye_ratio = eye_ratio(rightEye)
        left_eye_ratio = eye_ratio(leftEye)
        ear = (left_eye_ratio + right_eye_ratio) / 2.0
        rightEyeHull = cv2.convexHull(rightEye)
        leftEyeHull = cv2.convexHull(leftEye)
        cv2.drawContours(img, [leftEyeHull], -1, (212, 66, 244), 1)
        cv2.drawContours(img, [rightEyeHull], -1, (212, 66, 244), 1)
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                cv2.putText(img, "WAKE UP!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                alert()
        else:
            COUNTER = 0
            ALARM_ON = False
        cv2.putText(img, "{:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("IMG", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()