# Blood Cells Detection and Counting Web App with Flask and YOLOv8

This repository provides a user-friendly web application using YOLOv8, a powerful object detection model from Ultralytics https://www.ultralytics.com/. The app is built with Flask, a lightweight web framework for Python https://flask.palletsprojects.com/.


## Project Structure
```
YOLOv8-Flask-Web-App/

├── app.py                  # Main Flask application script
├── requirements.txt        # List of required Python dependencies
├── README.md
├── templates/
│   ├── index.html           # HTML template for the web interface
│   └── ... (other templates) # Optional additional templates
├── models/
│   └── yolov8_model.pt       # Trained YOLOv8 model weights file
└── static/
    ├── css/                 # CSS styles for the web interface
    │   └── ...
```
## Installation

### Clone the repository:
```
Bash
git clone https://github.com/12mustafashahid/YOLOv8-Flask-Web-App.git
```

### Create a virtual environment (recommended):
```
Bash
python -m venv venv
source venv/bin/activate  # Activate the virtual environment (Linux/macOS)
venv\Scripts\activate.bat  # Activate the virtual environment (Windows)
```

### Install dependencies:
```
Bash
pip install -r requirements.txt
```

### Running the Application

Start the Flask development server:
```
Bash
python app.py
```

Open your web browser:

```Navigate to http://127.0.0.1:5000/ (or the default Flask development server address) to access the web app.```
