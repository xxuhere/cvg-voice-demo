Raspberry Pi 5 setup for dual_camera_overlay_nira_sherpa(1).py
================================================================

Updated: 2026-10-06

PURPOSE
-------
This guide prepares a Raspberry Pi 5 to run:

    dual_camera_overlay_nira_sherpa(1).py

The program uses two IMX219 cameras, Picamera2, PyQt5/OpenGL, Sherpa-ONNX
keyword spotting, a microphone, and whisper.cpp. All voice processing runs
locally after the required software and model files have been downloaded.

The current wake word is spoken as three letters: "C V G". After it is
recognized, the GUI flashes a green LISTENING indicator. There is deliberately
no audible beep.


1. HARDWARE AND OPERATING-SYSTEM ASSUMPTIONS
--------------------------------------------

Recommended:

* Raspberry Pi 5 with current 64-bit Raspberry Pi OS Bookworm or later.
* Two Raspberry Pi Camera Module v2/IMX219 cameras connected to the two Pi 5
  MIPI camera connectors.
* A desktop/graphical session. The program displays a PyQt5/OpenGL window.
* Adequate Pi 5 cooling and the official 27 W power supply or equivalent.
* An SSD with sufficient free space for two simultaneous MP4 recordings.
* A microphone recognized by PipeWire/ALSA. A USB microphone is normally more
  reliable than a Bluetooth earbud microphone.

Shut down and remove power before connecting or disconnecting camera cables.


2. UPDATE RASPBERRY PI OS
-------------------------

Run:

    sudo apt update
    sudo apt full-upgrade -y
    sudo reboot

After reboot, open a terminal in the graphical desktop.


3. INSTALL SYSTEM PACKAGES
--------------------------

Run:

    sudo apt update
    sudo apt install -y \
        python3-full python3-venv python3-pip python3-numpy \
        python3-picamera2 python3-pyqt5 python3-av \
        libportaudio2 portaudio19-dev \
        git cmake build-essential wget bzip2 ffmpeg

Optional microphone/profile control GUI:

    sudo apt install -y pavucontrol

Important:

* Install Picamera2 from Raspberry Pi OS with apt, not pip.
* Do not use the obsolete legacy Picamera package or raspistill/raspivid stack.
* The virtual environment created below uses --system-site-packages so it can
  see the apt-installed Picamera2, PyQt5, NumPy, and PyAV packages.

Official Raspberry Pi camera documentation:

    https://www.raspberrypi.com/documentation/computers/camera_software.html


4. VERIFY BOTH CAMERAS BEFORE RUNNING PYTHON
--------------------------------------------

List detected cameras:

    rpicam-hello --list-cameras

The output must show two cameras, normally indices 0 and 1. Test each camera:

    rpicam-hello --camera 0 --timeout 3000
    rpicam-hello --camera 1 --timeout 3000

The Python script assumes:

    Camera 0 = NIR/reference IMX219 NoIR camera
    Camera 1 = visible IMX219 camera

If the order is reversed, swap the camera connections or change these constants
near the top of the Python file:

    NIR_CAMERA_INDEX = 0
    VISIBLE_CAMERA_INDEX = 1

The Pi 5 NoIR tuning file used by the script should exist here:

    /usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json

Check it with:

    test -f /usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json \
        && echo "NoIR tuning file found" \
        || echo "NoIR tuning file not found"


5. PLACE THE PYTHON SCRIPT AND HOMOGRAPHY FILE
----------------------------------------------

The examples in this guide use:

    /home/pi/Camera/

Create it if necessary:

    mkdir -p /home/pi/Camera

Place the Python script in that directory:

    /home/pi/Camera/dual_camera_overlay_nira_sherpa(1).py

The script requires this file before the cameras can start:

    /home/pi/Camera/homography.npy

The 3 x 3 matrix must map visible-camera pixels into NIR-camera pixels. Use the
calibrated matrix for aligned overlay operation.

For an initial software test only, create an identity matrix:

    python3 - <<'PY'
    import numpy as np
    np.save("/home/pi/Camera/homography.npy", np.eye(3, dtype=np.float64))
    print("Created identity homography for testing")
    PY

An identity matrix does not align cameras having different viewpoints. Replace
it with the calibrated homography for actual CVG use. The GUI also provides a
Select Homography button.


6. CREATE THE PYTHON VIRTUAL ENVIRONMENT
----------------------------------------

Raspberry Pi OS protects its system Python and may report an
"externally-managed-environment" error if pip is used globally. Use a virtual
environment instead:

    python3 -m venv --system-site-packages /home/pi/cvg-voice-env
    source /home/pi/cvg-voice-env/bin/activate
    python -m pip install --upgrade pip
    python -m pip install sherpa-onnx sounddevice sentencepiece

Activate this environment every time before starting the program:

    source /home/pi/cvg-voice-env/bin/activate

Do not use sudo with pip inside the virtual environment.


7. DOWNLOAD THE SHERPA-ONNX ENGLISH KEYWORD MODEL
-------------------------------------------------

The required model is:

    sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01

Download and extract it next to the Python script:

    cd /home/pi/Camera
    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
    tar xvf sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
    rm sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2

Verify the folder:

    ls -lh /home/pi/Camera/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01

It must contain all of the following:

    bpe.model
    tokens.txt
    encoder-*.onnx
    decoder-*.onnx
    joiner-*.onnx

The program automatically prefers the INT8 ONNX files when they are present.

Official Sherpa-ONNX keyword-model documentation:

    https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html


8. BUILD WHISPER.CPP AND DOWNLOAD A WHISPER MODEL
-------------------------------------------------

Clone and build whisper.cpp:

    cd /home/pi
    git clone https://github.com/ggml-org/whisper.cpp.git
    cd /home/pi/whisper.cpp
    cmake -B build
    cmake --build build -j4 --config Release

Download the small English model:

    cd /home/pi/whisper.cpp
    sh ./models/download-ggml-model.sh tiny.en

Expected files:

    /home/pi/whisper.cpp/build/bin/whisper-cli
    /home/pi/whisper.cpp/models/ggml-tiny.en.bin

Verify them:

    test -x /home/pi/whisper.cpp/build/bin/whisper-cli \
        && echo "whisper-cli found" \
        || echo "whisper-cli missing"

    test -f /home/pi/whisper.cpp/models/ggml-tiny.en.bin \
        && echo "Whisper model found" \
        || echo "Whisper model missing"

Test whisper.cpp independently:

    /home/pi/whisper.cpp/build/bin/whisper-cli \
        -m /home/pi/whisper.cpp/models/ggml-tiny.en.bin \
        -f /home/pi/whisper.cpp/samples/jfk.wav \
        -l en

For better command accuracy at the cost of longer recognition time, download
base.en instead:

    cd /home/pi/whisper.cpp
    sh ./models/download-ggml-model.sh base.en

Then use ggml-base.en.bin in WHISPER_MODEL_PATH.

Official whisper.cpp repository:

    https://github.com/ggml-org/whisper.cpp


9. CONFIGURE AND TEST THE MICROPHONE
------------------------------------

If using Bluetooth earbuds, select a headset profile that includes microphone
input (HSP/HFP), not an A2DP playback-only profile. This can be selected in
pavucontrol under Configuration and Input Devices.

Inspect PipeWire devices:

    wpctl status

If necessary, set the microphone source as the default using its PipeWire node
ID from the Sources section:

    wpctl set-default PIPEWIRE_SOURCE_NODE_ID
    wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0
    wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 1.0

Test recording and playback:

    pw-record /home/pi/cvg_mic_test.wav

Speak normally for several seconds, then press Ctrl+C. Play it:

    pw-play /home/pi/cvg_mic_test.wav

The recording should be clear and undistorted.

While the virtual environment is active, list PortAudio/sounddevice devices:

    source /home/pi/cvg-voice-env/bin/activate
    python -m sounddevice

Choose an entry that has at least one input channel. An asterisk marks the
current default device. For the previously tested Pi configuration, device 4
was the default input, but device numbers can change after rebooting, pairing,
or reconnecting hardware.

Important:

* CVG_AUDIO_DEVICE uses the number printed by "python -m sounddevice".
* It does NOT use the PipeWire node ID printed by "wpctl status".
* Set CVG_AUDIO_DEVICE=-1 to let sounddevice use the current default input.


10. SET THE REQUIRED ENVIRONMENT VARIABLES
------------------------------------------

Run these commands in the same terminal that will launch the program:

    export SHERPA_KWS_MODEL_DIR="/home/pi/Camera/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    export WHISPER_CLI_PATH="/home/pi/whisper.cpp/build/bin/whisper-cli"
    export WHISPER_MODEL_PATH="/home/pi/whisper.cpp/models/ggml-tiny.en.bin"

Set the microphone device after checking "python -m sounddevice":

    export CVG_AUDIO_DEVICE=4

Recommended starting voice settings:

    export CVG_COMMAND_SECONDS=5.0
    export CVG_KWS_THRESHOLD=0.20
    export CVG_KWS_SCORE=2.0

Notes:

* Lower CVG_KWS_THRESHOLD values make wake-word activation easier but may
  increase false activations. Try 0.15 to 0.30.
* CVG_KWS_SCORE changes keyword preference. The recommended starting value is
  2.0.
* CVG_COMMAND_SECONDS controls how long command audio is recorded after wake
  detection.
* Legacy NIRA_AUDIO_DEVICE, NIRA_COMMAND_SECONDS, NIRA_KWS_THRESHOLD, and
  NIRA_KWS_SCORE names remain accepted as fallbacks, but CVG_* is preferred.
* Environment variables set with export apply only to that terminal and its
  child processes. Starting the program separately from Thonny may not inherit
  them.


11. RUN A PYTHON IMPORT PREFLIGHT TEST
--------------------------------------

Activate the virtual environment and run:

    source /home/pi/cvg-voice-env/bin/activate
    python -c "import av, numpy, sentencepiece, sherpa_onnx, sounddevice; from picamera2 import Picamera2; from PyQt5.QtWidgets import QApplication; print('All required Python imports succeeded')"

If Picamera2, PyQt5, or av cannot be imported, confirm that the environment was
created with --system-site-packages.


12. START THE PROGRAM
---------------------

From the same graphical terminal where the environment variables were set:

    source /home/pi/cvg-voice-env/bin/activate
    python "/home/pi/Camera/dual_camera_overlay_nira_sherpa(1).py"

Suggested initial settings:

    Capture resolution:      1280x720
    Capture/record FPS:      30
    GPU overlay resolution:  1280x720
    Display target FPS:      24

For 1920x1080 capture and recording, 24 FPS is generally more stable than 30
FPS on this dual-camera application.


13. USING VOICE CONTROL
-----------------------

1. Click Enable CVG Voice.
2. Wait for:

       VOICE: READY - say 'C V G'

3. Say "C V G" as three letters.
4. When the green LISTENING badge flashes, immediately say one command.

Supported command groups include:

    start cameras
    stop cameras
    start recording
    stop recording
    capture raw pair
    increase overlay
    decrease overlay
    exit program
    confirm
    cancel

The script also recognizes several close aliases, such as "start camera" and
"capture image".

To exit by voice:

1. Say "C V G".
2. During LISTENING, say "exit program".
3. Within 10 seconds, say "C V G" again.
4. During LISTENING, say "confirm".


14. OUTPUT FILES
----------------

The default output directory is:

    /home/pi/Videos/dual_camera

It is created automatically when needed. The Select Output Folder button can
change it. Recording produces MP4 files and timestamp CSV files. Capture Raw
Bayer Pair produces DNG and NumPy NPY data for both cameras.

Ensure that the destination drive has enough free space:

    df -h


15. COMMON PROBLEMS
-------------------

"externally-managed-environment"
    Do not install the pip packages globally. Activate cvg-voice-env and run
    pip inside it.

"Sherpa KWS model folder not found"
    Correct SHERPA_KWS_MODEL_DIR or place the extracted model folder directly
    beside the Python script.

"Sherpa encoder/decoder/joiner/tokens.txt/bpe.model not found"
    The model archive is incomplete, the wrong model was downloaded, or the
    environment variable points to the wrong directory.

"whisper-cli not found"
    Correct WHISPER_CLI_PATH. The expected executable is normally:
    /home/pi/whisper.cpp/build/bin/whisper-cli

"Whisper model not found"
    Correct WHISPER_MODEL_PATH. Do not point it at the whisper.cpp source
    directory; point it at the actual ggml-*.bin file.

No microphone or PortAudio error
    Run "python -m sounddevice" in the active environment. Verify that the
    selected device has input channels and update CVG_AUDIO_DEVICE.

Bluetooth audio is audible but the microphone is missing
    Change the Bluetooth profile from A2DP to HSP/HFP. A2DP is normally
    playback-only.

Wake word works only when spoken loudly
    Confirm microphone gain and clarity, then try CVG_KWS_THRESHOLD=0.15.
    Lower values may increase false activations. The generic Sherpa model may
    also be sensitive to accent and pronunciation.

Wake word is detected but the command is not understood
    Speak only after the green LISTENING badge begins flashing. Increase
    CVG_COMMAND_SECONDS to 5.0. Consider ggml-base.en.bin if tiny.en does not
    recognize the operator reliably.

No audible wake beep
    This is expected. The Qt beep was intentionally removed because it was not
    reliably routed on Raspberry Pi Linux. Use the flashing green LISTENING
    indicator.

"Homography file not found"
    Put homography.npy beside the script or choose it with Select Homography.
    Use the identity-matrix instructions in Section 5 only for initial testing.

Only one camera is listed
    Power down, reseat both MIPI cables, verify cable orientation, reboot, and
    rerun "rpicam-hello --list-cameras" before starting Python.

OpenGL/GUI errors over SSH
    Run the application from the Pi graphical desktop. X forwarding is not a
    reliable configuration for the camera/OpenGL preview.

Camera recording or display becomes slow
    Begin testing at 1280x720, 30 FPS capture, and 24 FPS display. Use 24 FPS
    capture for 1920x1080. Check cooling, throttling, free disk space, and SSD
    write performance.


16. COMPLETE PRE-LAUNCH CHECKLIST
---------------------------------

[ ] Current 64-bit Raspberry Pi OS is installed and updated.
[ ] Both IMX219 cameras appear in rpicam-hello --list-cameras.
[ ] Camera 0 is the intended NIR/reference camera.
[ ] Camera 1 is the intended visible camera.
[ ] Picamera2, PyQt5, PyAV, and NumPy imports succeed.
[ ] /home/pi/cvg-voice-env exists and is activated.
[ ] sherpa-onnx, sounddevice, and sentencepiece imports succeed.
[ ] The Sherpa GigaSpeech KWS folder contains ONNX files, tokens.txt, and
    bpe.model.
[ ] whisper-cli exists and is executable.
[ ] The ggml-tiny.en.bin or ggml-base.en.bin file exists.
[ ] The microphone produces a clear test recording.
[ ] CVG_AUDIO_DEVICE matches the current sounddevice input number.
[ ] homography.npy exists or has been selected in the GUI.
[ ] The output SSD/folder has sufficient free space.
[ ] All export commands were run in the same terminal used to start Python.

Once every item is satisfied, the script should start, open its GUI, initialize
both cameras, and enable the offline CVG voice-control pipeline.

17. Summary added by Xiao
execute the following before starting the Python program, python "/home/pi/Camera/dual_camera_overlay_nira_sherpa(2).py"

    source /home/pi/cvg-voice-env/bin/activate
    export SHERPA_KWS_MODEL_DIR="/home/pi/Camera/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    export WHISPER_CLI_PATH="/home/pi/whisper.cpp/build/bin/whisper-cli"
    export WHISPER_MODEL_PATH="/home/pi/whisper.cpp/models/ggml-tiny.en.bin"
    export CVG_AUDIO_DEVICE=1
    export CVG_COMMAND_SECONDS=3.5
    export CVG_KWS_THRESHOLD=0.20
    export CVG_KWS_SCORE=2.0
    python "/home/pi/Camera/dual_camera_overlay_nira_sherpa(2).py"