# Phishing Website Detection System

An advanced AI-powered system for detecting phishing websites using machine learning, a Flask API, and a Chrome extension.

## Overview

This project is a comprehensive phishing detection solution that consists of three main components:

1. **Machine Learning Model**: An XGBoost classifier trained to identify phishing websites based on URL and website features.
2. **Flask API**: A RESTful API that serves the model predictions and provides feature importance information.
3. **Chrome Extension**: A user-friendly browser extension that allows users to scan websites in real-time.

## Project Structure

```
/
├── model/                    # Machine learning model code
│   ├── train_model.py        # Script to train the XGBoost model
│   ├── feature_extraction.py # Feature extraction utilities
│   └── saved/                # Directory for saved model files
│
├── api/                      # Flask API code
│   ├── app.py                # Main API application
│   ├── templates/            # HTML templates for API documentation
│   └── static/               # Static assets for API documentation
│
├── extension/                # Chrome extension code
│   ├── manifest.json         # Extension manifest file
│   ├── popup.html            # Extension popup UI
│   ├── popup.js              # Extension popup logic
│   ├── styles.css            # Extension styling
│   ├── background.js         # Extension background script
│   └── icons/                # Extension icons
│
├── data/                     # Data directory
│   ├── phishing_dataset.csv  # Training/testing dataset
│   └── feature_importance.csv# Feature importance data
│
└── src/                      # React web application
    ├── App.tsx               # Main application component
    ├── components/           # React components
    ├── index.css             # Global styles
    └── main.tsx              # Application entry point
```

## Features

- **URL Analysis**: Extracts features from URLs such as length, special characters, domain information, etc.
- **Domain Analysis**: Checks domain age, SSL certificates, suspicious TLDs, etc.
- **HTML/JS Analysis**: Examines page content for forms, iframes, and other potentially suspicious elements.
- **Real-time Detection**: Analyzes websites as users browse them.
- **Detailed Reports**: Provides comprehensive reports on why a website was flagged.
- **Visualization**: Shows feature importance and model performance metrics.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Node.js and npm
- Chrome browser

### Setting Up the Model

1. Install required packages:
   ```
   pip install pandas numpy scikit-learn xgboost joblib
   ```

2. Train the model:
   ```
   python model/train_model.py
   ```

### Starting the Flask API

1. Install required packages:
   ```
   pip install flask flask-cors
   ```

2. Start the API server:
   ```
   python api/app.py
   ```

3. The API will be available at `http://localhost:5000`

### Installing the Chrome Extension

1. Open Chrome and navigate to `chrome://extensions`
2. Enable "Developer mode" by toggling the switch in the top-right corner
3. Click "Load unpacked" and select the `extension` directory
4. The PhishGuard extension should now appear in your extensions list

### Running the Web Application

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

## Using the Extension

1. Navigate to any website you want to check
2. Click the PhishGuard icon in your browser toolbar
3. Click the "Scan Website" button in the popup
4. Review the results and, if desired, click "View Details" for more information

## Model Performance

The XGBoost model achieves:
- Accuracy: ~94%
- Precision: ~92%
- Recall: ~95%
- F1 Score: ~93%

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The XGBoost team for their excellent machine learning library
- Flask for the simple and powerful web framework
- The Chrome Extensions API documentation