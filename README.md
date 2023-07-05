# MLOps-Marathon-2023

## Prerequisites
* Python 3.10.0: The programming language used for this application.

## Getting Started
1. Clone the repository to your local machine using the following command:
```
git clone https://github.com/tdbui1209/MLOps-Marathon-2023.git
```
2. Change into the project directory:
```
cd MLOps-Marathon-2023
```
3. Create a virtual environment to isolate the application's dependencies:
```
python3 -m venv venv
```
4. Activate the virtual environment:
```
venv\Scripts\activate
```
5. Install the required dependencies:
```
pip install -r requirements.txt
```
6. Run the application:
To specify custom host and port values, use the --host and --port arguments:
```
python flask_app.py --host 0.0.0.0 --port 8080
```
If you want to capture request data, use the --capture-data argument:
```
python flask_app.py --capture-data True
```

## Project Structure
The project's directory structure is organized as follows:
```php
MLOps-Marathon-2023/
  ├── config/                    # Config of each problems
  ├── data/                      # Store captured_data, raw_data, training_data
  ├── models/                    # Store models for preprocess and predict data
  ├── data_processor.py          # Process the input data
  ├── flask_app.py               # Entry point of the Flask application
  ├── predictor.py               # Predict data and posprocessing
  ├── requirements.txt           # List of dependencies required by the application
  └── utils.py                   # AppPath contains the path of the project
```
