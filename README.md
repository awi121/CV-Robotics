# CV-Robotics

## Getting Started
1. Clone repository
2. Create virtual environment (Python 8)
    * Mac `python -m venv venv`
    * Windows `py -3.8 -m venv venv`
3. Enter virtual environment
    * Mac `source ./venv/bin/activate`
    * Windows `source ./venv/Scripts/Activate`
4. Install packages `pip install -r requirements.txt`
5. Install PyRealsense2
    * Mac `pip install pyrealsense2-macosx`
    * Windows `pip install pyrealsense2`

## Notes
* `capture.py` handles writing `.ply` and `.png` files
* `processing.py` handles the processing of files into point clouds.
