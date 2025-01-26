# California Housing Price Prediction MLOps Project

![CI/CD](https://github.com/[username]/california-housing-mlops/workflows/CI/CD%20Pipeline/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

A production-ready machine learning project demonstrating MLOps best practices using California housing price prediction as an example.

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features
- Machine learning pipeline for housing price prediction
- RESTful API for model serving
- MLflow for experiment tracking
- Containerized deployment with Docker
- Automated testing and CI/CD pipeline
- Code quality checks and formatting
- Comprehensive documentation

## 📁 Project Structure
california-housing-mlops/ 
├── .github/  
|	└── workflows/  
|		├── ci-cd.yml 
|		└── release.yml 
├── src/ 
|	├── init.py  
|	├── model.py 
│ 	├── data.py 
│ 	├── train.py 
│ 	└── predict.py 
├── tests/ 
│ 	├── init.py 
│ 	└── test_model.py 
├── models/
|	├── model_20240120_123456/
│   		└── model.pkl
|	├── model_20240120_123789/
│   		└── model.pkl
|	└── latest -> model_20240120_123789/ 
├── notebooks/ 
├── requirements.txt 
├── requirements-dev.txt 
├── Dockerfile 
├── .gitignore 
└── README.md

## 📋 Requirements
- Python 3.12+
- Docker (for containerization)
- Make (optional, for using Makefile commands)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/california-housing-mlops.git
cd california-housing-mlops
Create and activate virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
# For production
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
Install pre-commit hooks (for development):
pre-commit install
💻 Usage
Training the Model
python -m src.train
This will:

Load the California housing dataset
Train a Random Forest model
Save the model to the models directory
Log metrics and parameters to MLflow
Starting the API
python -m src.predict
The API will be available at http://localhost:5000

📚 API Documentation
Get Model Information
curl http://localhost:5000/info
Response:

{
    "feature_names": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
    "description": "California Housing Price Prediction Model",
    "feature_descriptions": {
        "MedInc": "Median income in block group",
        "HouseAge": "Median house age in block group",
        ...
    }
}
Make Predictions
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]}'
Response:

{
    "status": "success",
    "prediction": 452000.0
}

# 🛠 Development
## Using Make Commands
# Set up development environment
make setup

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Clean up temporary files
make clean
Branch Strategy
Create a feature branch:
git checkout -b feature/your-feature-name
Make changes and commit:
git add .
git commit -m "feat: add your feature"
Push changes and create a pull request:
git push origin feature/your-feature-name
# 🧪 Testing

## Run the test suite:

pytest tests/

##Run with coverage:

pytest tests/ --cov=src --cov-report=term-missing
📦 Deployment
Using Docker
Build the image:
docker build -t california-housing-model .
Run the container:
docker run -p 5000:5000 california-housing-model
Using Docker Compose (for development)
docker-compose up --build
## 🤝 Contributing
Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'feat: add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
California Housing dataset from scikit-learn
MLflow for experiment tracking
All contributors and maintainers
For more information or questions, please open an issue or contact [your-email@example.com].



