# tello-gcs

[Work in progress]

Various applications for the Tello Drone

***

Requirements:

Tello EDU Drone
Firmware version: 2.05 +

***

Only extra install needed is tesseract executable:

For Windows: https://github.com/UB-Mannheim/tesseract/wiki

For Mac/Linux: https://tesseract-ocr.github.io/tessdoc/Installation.html#macos


***


Credits to:
https://github.com/geaxgx/tello-openpose

For creating the body control script for the tello drone. 
I used almost all of his main code to create my version of the body control. Except I used the mediapipe python library instead of openpose to detect body landmarks. (It is easier to install and I can use the same library for face recognition as well).
