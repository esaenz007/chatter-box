#!/bin/bash
xset s off       # disable screensaver
xset -dpms       # disable power management
xset s noblank   # disable blanking

export OPENAI_API_KEY=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

source /home/pi/oldtv/venv/bin/activate

python3 /home/pi/oldtv/main.py >> /home/pi/kiosk.log 2>&1