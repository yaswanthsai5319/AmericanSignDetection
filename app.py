import cv2
from flask import Flask, render_template, Response,redirect,url_for,session
from flask_socketio import SocketIO, emit
from flask import Flask, render_template
import cv2 as cv
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark
import mediapipe as mp
import numpy as np
from model import KeyPointClassifier
import copy
import csv
import time

from ans import ans
app = Flask(__name__)
# app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
# socketio = SocketIO(app)

# @app.route('/')
# def index():
#     return render_template('index.html')


# @socketio.on('video_stream')
# def handle_video_stream(video_data):
#     # process the video data here
#     print("Received video stream")
#     print(type(video_data))
#     # Convert the video data into a numpy array
#     numpy_frame = np.frombuffer(video_data, np.uint8)
#     # Decode the numpy array into a cv2 image
#     frame = cv.imdecode(numpy_frame, cv.IMREAD_COLOR)
#     processed_data = hand_sign_recognition(frame)
#     # Emit the processed data back to the client
#     emit('processed_data', processed_data)


# def hand_sign_recognition(frame):
#     gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     return gray_frame
#     args = get_args()
#     cap_device = args.device
#     cap_width = args.width
#     cap_height = args.height
#     use_static_image_mode = args.use_static_image_mode
#     min_detection_confidence = args.min_detection_confidence
#     min_tracking_confidence = args.min_tracking_confidence
#     img = cv.VideoCapture(0)
#     img.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
#     img.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=use_static_image_mode,
#         max_num_hands=1,
#         min_detection_confidence=min_detection_confidence,
#         min_tracking_confidence=min_tracking_confidence,
#     )
#     keypoint_classifier = KeyPointClassifier()

#     with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
#         keypoint_classifier_labels = csv.reader(f)
#         keypoint_classifier_labels = [
#             row[0] for row in keypoint_classifier_labels
#         ]
#     while True:
#         key = cv.waitKey(10)
#         if key == 27:  # ESC
#             break

#         ret, image = img.read()
#         if not ret:
#             break
#         image = cv.flip(image, 1)
#         debug_image = copy.deepcopy(image)
#         # print(debug_image.shape)
#         # cv.imshow("debug_image",debug_image)
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True
#         if results.multi_hand_landmarks is not None:
#             for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                 landmark_list = calc_landmark_list(debug_image, hand_landmarks)
#                 pre_processed_landmark_list = pre_process_landmark(
#                     landmark_list)

#                 hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

#                 debug_image = draw_landmarks(debug_image, landmark_list)
#                 debug_image = draw_info_text(
#                     debug_image,
#                     handedness,
#                     keypoint_classifier_labels[hand_sign_id])
#         return debug_image
#     img.release()


# if __name__ == '__main__':
#     socketio.run(app)


camera = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    start_time=time.time()
    while True:
        if time.time()-start_time>14:
            print(time.time())
            # Capture frame-by-frame
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                frame = process(frame)
                ret, buffer = cv2.imencode('.jpg', frame)
                # print(buffer)
                frame = buffer.tobytes()
                # print(frame)
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                start_time=time.time()


def process(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)
    print(debug_image.shape)
    # cv.imshow("debug_image",debug_image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)

            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            print("hand",hand_sign_id)

            debug_image = draw_landmarks(debug_image, landmark_list)
            debug_image = draw_info_text(
                debug_image,
                handedness,
                keypoint_classifier_labels[hand_sign_id])
    else:
        if len(ans[0])>0 and ans[0][-1] != " ":
            ans[0]+=" "
    return debug_image


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    return redirect('/')

# @app.route('/new_page')
# def new_page():
#     return render_template('index.html',answer=ans[0])
@app.route('/')
def index():
    temp=ans[0]
    ans[0]=""
    return render_template('index.html',answer=temp)


if __name__ == '__main__':
    app.run(debug=True)
