#!/bin/bash
xset s off       # disable screensaver
xset -dpms       # disable power management
xset s noblank   # disable blanking

export OPENAI_API_KEY=

source /home/pi/oldtv/venv/bin/activate

python3 /home/pi/oldtv/main.py >> /home/pi/kiosk.log 2>&1