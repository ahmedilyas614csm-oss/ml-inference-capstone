# Real-Time ML Inference REST API

A production-ready REST API that serves a trained Machine Learning model for real-time predictions, built with FastAPI and containerized with Docker.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Tests](https://img.shields.io/badge/Tests-4%20Passed-brightgreen)

## 📖 Overview

This project packages a trained **Random Forest classifier** (trained on the Iris dataset) into a production-grade REST API. It demonstrates the complete MLOps workflow from model training to containerized deployment.

**Key Features:**
- 🚀 FastAPI microservice with automatic Swagger documentation
- 🤖 Real-time predictions with probability scores
- 📦 Fully containerized with Docker
- ✅ Comprehensive unit tests with pytest
- 🔒 Input validation via Pydantic schemas

## 🏗️ System Architecture
┌─────────────┐ JSON ┌─────────────┐ NumPy ┌─────────────┐
│ Client │ ───────────────▶│ FastAPI │ ──────────────▶│ ML Model │
│ (Browser) │ ◀───────────────│ /predict │ ◀──────────────│ (model.pkl)│
└─────────────┘ JSON Response └─────────────┘ Prediction └─────────────┘

**End-to-End Flow:**
1. Client sends a POST request with JSON features to `/predict`
2. FastAPI validates the input using Pydantic schemas
3. The trained model generates a prediction and probability
4. API returns structured JSON response

## 📁 Project Structure
ml-inference-capstone/
├── main.py # FastAPI application & endpoints
├── train_model.py # Script to train and save the model
├── model.pkl # Trained ML model (Random Forest)
├── test_main.py # Unit tests for the API
├── requirements.txt # Pinned Python dependencies
├── Dockerfile # Container configuration
├── .dockerignore # Files excluded from Docker build
└── README.md # Project documentation (this file)

## 🛠️ Tech Stack

| Component          | Technology            |
|--------------------|-----------------------|
| API Framework      | FastAPI               |
| Input Validation   | Pydantic              |
| ML Library         | scikit-learn          |
| Model Persistence  | joblib                |
| Web Server         | Uvicorn               |
| Containerization   | Docker                |
| Testing            | pytest, httpx         |
| Language           | Python 3.12           |

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker Desktop (optional, for containerized deployment)
- Git

### Installation (Local Setup)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/ml-inference-capstone.git
   cd ml-inference-capstone