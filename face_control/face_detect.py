import mediapipe as mp
import cv2
import csv
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score  # Accuracy metrics
import pickle
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def create_csv(csv_file: str):
    """
    Function to create CSV file with necessary colums (501 for face and body landmarks)

    :param csv_file: Name of the file
    :return: Name of the file
    """
    num_coords = 501  # (33 body landmarks + 468 face landmarks)

    # Rellenar los titulos de las columnas
    landmarks = ['class']
    for val in range(1, num_coords + 1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

    return csv_file


def add_class(csv_file: str, class_name: str, video_input: str):
    """
    Function to CREATE new class in the csv file or ADD data to existing class

    :param video_input: Choose if phone or webcam
    :param csv_file: Name of the file
    :param class_name: Name of the class that you want to add data to
    :return: None
    """
    mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
    mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

    # Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
    url = "http://192.168.0.11:8080/shot.jpg"

    if video_input == "webcam":
        # Takes video camera input to fill file
        cap = cv2.VideoCapture(0)

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            if video_input == "phone":
                img_resp = requests.get(url)
                img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
                frame = cv2.imdecode(img_arr, -1)

            if video_input == "webcam":
                ret, frame = cap.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(
                    np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                # Concatenate rows
                row = pose_row + face_row

                # Append class name
                row.insert(0, class_name)

                # Export to CSV
                with open(csv_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

            except:
                pass

            cv2.putText(image, f"Start and keep posing for '{class_name}'!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)
            cv2.putText(image, "To exit, press 'q' and wait patiently", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2)
            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    if video_input == "webcam":
        cap.release()

    cv2.destroyAllWindows()

    return None


def read_and_process_data(csv_file: str):
    """
    We need to train data in the CSV file. This function creates the training and test
     partitions needed to do this.

    :param csv_file: Name of the file
    :return: x_train, y_train --> To train the model. x_test, y_test --> to evaluate the model
    """
    df = pd.read_csv(csv_file)
    X = df.drop('class', axis=1)  # features
    y = df['class']  # target value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    return X_train, X_test, y_train, y_test


def train_ML_classification_model(X_train, y_train):
    """

    :param X_train:
    :param y_train:
    :return:
    """
    pipelines = {
        # 'lr': make_pipeline(StandardScaler(), LogisticRegression()),
        'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
        'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train.values, y_train)
        fit_models[algo] = model

    return fit_models


def eval_and_serialize_model(fit_models: dict, X_test, y_test):
    """
    Makes evaluation of the models to see what fits best, this function takes a while to execute.
    It creates a binary file containing the evaluated models ready to use

    :param fit_models: dictionary containing the models
    :param X_test: parameter to test the model
    :param y_test: parameter to test the model
    :return: a string of the binary file
    """
    binary_file = "../modules/body_language.pkl"
    for algo, model in fit_models.items():
        yhat = model.predict(X_test.values)
        print(algo, accuracy_score(y_test, yhat))

    with open(binary_file, 'wb') as f:
        pickle.dump(fit_models['rf'], f)

    return binary_file


def get_only_binary_file(fit_models: dict):
    binary_file = "../modules/body_language.pkl"
    with open(binary_file, 'wb') as f:
        pickle.dump(fit_models['rf'], f)

    return binary_file


def main(evaluate: bool, video_input: str, existing_csv_file=None, class_name=None, new_csv_file=False, add_class_only=False):
    if add_class_only:
        """Only adds data to the csv file to make it faster and load many data at once"""
        # Add data to existing file
        csv_file = existing_csv_file

        # Add a class
        add_class(csv_file, class_name, video_input)
        print(f"Class {class_name} was added to {csv_file} without updating the binary file")

    elif not class_name:
        """Only tests and trains the models if you previously only added data to the csv file"""
        print("Training models...")
        # Only train and test data
        csv_file = existing_csv_file
        # Read csv file and process data to get required parameters for the ML model
        X_train, X_test, y_train, y_test = read_and_process_data(csv_file)

        # Train our model
        fit_models = train_ML_classification_model(X_train, y_train)

        if evaluate:
            # Evaluate the effectiveness of our data and get the binary file
            binary_file = eval_and_serialize_model(fit_models, X_train, y_train)

        else:
            # Only get binary file
            binary_file = get_only_binary_file(fit_models)

        print(f"The binary file {binary_file} was created")

    else:
        """Add class and train and test at the same time"""
        if new_csv_file:
            # Create file from scratch
            csv_file = create_csv("../modules/coords.csv")

            # Add a class to new csv file
            add_class(csv_file, class_name, video_input)
            print(f"Class {class_name} added to {csv_file}")

        else:
            # Add data to existing file
            csv_file = existing_csv_file

            # Add a class
            add_class(csv_file, class_name, video_input)
            print(f"Class {class_name} added to {csv_file} \n")
            print("Training models...")

            # Read csv file and process data to get required parameters for the ML model
            X_train, X_test, y_train, y_test = read_and_process_data(csv_file)

            # Train our model
            fit_models = train_ML_classification_model(X_train, y_train)

            if evaluate:
                # Evaluate the effectiveness of our data and get the binary file
                binary_file = eval_and_serialize_model(fit_models, X_train, y_train)

            else:
                # Only get binary file
                binary_file = get_only_binary_file(fit_models)

            print(f"The binary file {binary_file} was created")


if __name__ == "__main__":
    """
    
    new_csv_file -> If true, parameter (existing_csv_file) True if first time running code, or if you want to delete old file and create new one.
    evaluate -> True if you want to evaluate the models (takes a long time).
    class_name -> Name of the class that you want to add data to.
    
    
            How to use:
                After executing the code, a cv2 window will pop up and you have to start posing and maintain the pose 
                as long as you want. The more you hold the pose, the more data the model will have but the more 
                will the code take to create the binary file. 
                
                To exit, press "q"
                
                You need at least 2 classes to create the binary file.
    
    
    Existing classes:
        - Normal (Normal straight face)
        - Sonriendo (Big smile, showing teeth)
        
    """

    """Examples of use"""
    # If creating new csv file or delete existing and start over (remember that you need at least 2 classes)
    # main(new_csv_file=True, evaluate=False, class_name="Normal", video_input="phone")

    # If csv file already exists
    # main(existing_csv_file="../modules/coords.csv", evaluate=False, class_name="Normal",
    #      add_class_only=False, video_input="phone")

    # To just add data to the csv and update binary file later (to make it faster)
    # main(existing_csv_file="../modules/coords.csv", evaluate=False, class_name="Sonriendo",
    #      add_class_only=True, video_input="phone")

    # To just train and test the models and get the binary files (if you have previously added a class only)
    main(existing_csv_file="../modules/coords.csv", evaluate=False, add_class_only=False, video_input="phone")

