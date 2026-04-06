# Animal Vision Application

This connects a machine learning model to a locally hosted Flask Web Interface.

## Project Structure
- `model_classifier.py`: Connects and trains the MobileNetV2 architecture.
- `animal_classifier_model.h5`: The finalized compiled Neural Network.
- `animal_vision_app/`: Application code
  - `app.py`: Flask webserver routes
  - `templates/` & `static/`: Frontend interface

## Get Started
1. Install dependencies:
`pip install -r requirements.txt`

2. CD into the inner application directory:
`cd animal_vision_app`

3. Start Flask Web Server
`python app.py`

*Default address: `http://127.0.0.1:5001`*
