# **AutoML Web Application using PyCaret and Flask**


## 📄 Project Overview


- This project is a Flask-based AutoML web application that allows users to:
- Upload a CSV dataset
- Automatically train machine learning models (classification or regression)
- Compare different models and select the best one
- Save the best-performing model
- View model performance leaderboard
- Chat with an FAQ-based assistant (fallback to Gemini AI for complex queries)
- It uses PyCaret to automate the model building process and Google Gemini 1.5 Pro for enhanced chatbot responses.

## contributers
- 👤 SALMAN (ME)
- 👤 VISHNU SURESH
- 👤 NANDANA
- 👤 ADHITYAN 

Big thanks to my teammates for their incredible contributions throughout the project! 🙌

## 🚀 Features

- 📂 Upload any structured dataset (CSV format)
- 🎯 Select target column and problem type (classification or regression)
- ⚡ Automatic model training, evaluation, and feature selection
- 🏆 Leaderboard showing model performances
- 💾 Save the best model for future use
- 🤖 Intelligent chatbot for project-related queries

## 🛠️ Tech Stack

- Backend: Python, Flask
- AutoML: PyCaret
- Chatbot: Google Gemini 1.5 Pro API
- Frontend: HTML, Bootstrap
- Others: Pandas, Difflib, Dotenv


## 🧠 How It Works
- Upload CSV ➔ Choose target_column and problem_type ➔ AutoML starts
- PyCaret runs multiple models and selects the best based on performance.
- Leaderboard shows the ranking of all trained models.
- Model is saved in static/models/.
- Chatbot first searches FAQs; if no close match, Gemini generates the answer.

## 📝 Example Usage
- Upload a dataset in csv form.
- Select the target column.
- Choose problem type.
- View the best model, selected features, and the model leaderboard!
- Allow downloading the trained model (.pkl file)

## 📚 Requirements
- Python 3.8+
- Flask
- PyCaret
- pandas
- google-generativeai
- python-dotenv

See requirements.txt for full details.

## 🤖 Future Improvements

- Add model explainability (SHAP values, feature importance plots)
- Cloud deployment

## ✨ Credits
- PyCaret - Low-code machine learning library
- Flask - Web framework
- Google Gemini - Generative AI


## 📬 Contact
For queries or collaborations:

- Name: SALMAN .P 
- Email: salmanpayyanakkal3@gmail.com
- GitHub: mohmdsalman
