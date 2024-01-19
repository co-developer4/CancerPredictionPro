# Cancer Risk Prediction Django Application

This repository contains a Django web application that uses an Artificial Neural Network (ANN) to predict cancer risk levels based on various health and lifestyle factors. The project allows users to input their data, receive a risk assessment, and download the trained model for further study.

## Features

- Data input through a user-friendly web interface.
- Real-time cancer risk level prediction using a trained ANN model.
- Option to download the trained model for offline analysis.

## Installation

To set up this project on your local machine, follow these steps:

1. **Clone the Repository:**

   ```
   git clone https://github.com/co-developer4/CancerPredictionPro.git
   cd CancerPredictionPro
   ```

2. **Set up a Virtual Environment (Optional but recommended):**

   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```
   pip install -r requirements.txt
   ```

4. **Set up the Database:**

   ```
   python manage.py migrate
   ```

5. **Run the Development Server:**

   ```
   python manage.py runserver
   ```

   The application will be available at `http://localhost:8000/`.

## Usage

After starting the server, you can:

- Navigate to the main page to input the required data for prediction.
- Submit the data to get an immediate risk assessment.
- And then you will get a file from server. But it takes a few seconds.
