# Cybersecurity Audit Tool with Integrated AI

### Overview

This repository hosts a modular cybersecurity auditing system designed for SMEs.
It identifies vulnerabilities and explains them in simple, understandable language.

### Structure

- **frontend/** — React web interface for users to input website URLs and view reports.
- **backend/** — Spring Boot server to coordinate scans and handle API requests.
- **ai_module/** — Python Flask microservice for AI-based risk scoring.
- **database/** — SQL for PostgreSQL storage.
- **docs/** — Documentation including project synopsis and design notes.

### AI Module Setup

To use the AI-driven risk scoring features of the Cybersecurity Audit Tool, you must first prepare and start the Python AI service.

1. Prerequisites
   Ensure you have Python 3.8 or higher installed on your system. You can verify your Python version by running the following command in your terminal:
   python3 --version
2. Install Dependencies
   Navigate to the ai_module directory and install the required Python libraries using the provided requirements file:
   cd ai_module
   pip install -r requirements.txt
3. Train the AI Model
   The AI model must be trained on the initial dataset before it can make predictions. This process teaches the system how to recognize and score different vulnerability types. Run the training script:
   python3 model_trainer.py
   Wait for the confirmation message "Model training completed successfully!" to appear. This creates the trained model file required for the API to function.
4. Start the AI API Server
   Once the model is trained, start the Flask API server. This allows the backend of the audit tool to communicate with the AI model:
   python3 app.py
   The server will run on http://127.0.0.1:5000. You must keep this terminal window open while using the audit tool to ensure risk scores can be generated in real-time.
5. Verify the Installation
   To confirm that the AI module is ready for use, you can test the health endpoint from a separate terminal window:
   curl http://127.0.0.1:5000/health
   A successful installation will return a JSON response confirming that the status is "healthy" and the model is "loaded".
