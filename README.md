# TELLO GROUND CONTROL STATION

[Work in progress]

Various applications for the Tello Drone. 
Working on making it compatible with macOs and Windows as well.

***
# Current bugs 

OpenCV:
When installing libraries with requirements.txt, it installs more than one opencv library, making it unusable.

Current (temporary) solution:

Delete ALL opencv libraries and then only re-install the one in the requirements.txt (That version is the only one that works, later versions also have bugs with the code)


***

# Requirements:

Tello EDU Drone

Firmware version: 2.05 +

For body_control_leds.py you need the Robomaster TT

***

Only extra install needed is tesseract executable (Working on adding the mac files to the project as well):

For Mac/Linux: https://tesseract-ocr.github.io/tessdoc/Installation.html#macos


***


Credits to:
https://github.com/geaxgx/tello-openpose
For the body control scrip.
