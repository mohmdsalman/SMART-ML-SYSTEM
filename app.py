# Import necessary libraries for Flask, file handling, model training, JSON manipulation, and integration with Google Gemini AI.
from flask import Flask, render_template, request, jsonify
from model_utils import train_model  # Your AutoML logic here (assumed to be in model_utils.py)
import os
import pandas as pd
import json
import difflib
import google.generativeai as genai  # Used for integrating with Google Gemini AI
from dotenv import load_dotenv  # Used for loading environment variables

# Function to format the summary of the training result
def format_training_summary(data):
    model = data.get("best_model", "N/A")  # Extract the best model from the result data (defaults to "N/A" if not found)
    features = data.get("selected_features", [])  # Get the selected features
    message = data.get("details", "")  # Get the details of the training process

    # Create an HTML list of the selected features
    features_list = "".join(f"<li>{f}</li>" for f in features)

    # Create an HTML summary of the training results
    summary_html = f"""
    <p><strong>Status:</strong> {message}</p>
    <p><strong>Best Model:</strong></p>
    <pre style='background-color:#f5f5f5;padding:10px;border-radius:6px;'>{model}</pre>
    <p><strong>Selected Features:</strong></p>
    <ul>{features_list}</ul>
    """
    return summary_html  # Return the formatted HTML summary

# Load environment variables from a .env file (for sensitive data like API keys)
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Configure the Gemini AI model with the API key stored in the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model for chat interaction
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Start a chat session with the Gemini model, storing the conversation history (initially empty)
chat = gemini_model.start_chat(history=[])

# Define the route for the home page
@app.route('/')
def index():
    # Render and return the home page template (index.html)
    return render_template('index.html')

# Define the route for handling model training
@app.route('/train', methods=['POST'])
def train():
    # Set up the folder to store uploaded files
    uploads_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(uploads_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Check if the file was part of the request
    if 'file' not in request.files:
        return "No file part in the request."

    # Get the uploaded file from the request
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return "No file selected for upload."

    # Save the uploaded file to the uploads folder
    file_path = os.path.join(uploads_folder, file.filename)
    file.save(file_path)

    # Check if the file exists at the specified path
    if not os.path.exists(file_path):
        return f"File not found at {file_path}."

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Get the target column and problem type from the form data
    target_column = request.form['target_column']
    problem_type = request.form['problem_type']

    # Train the model using the train_model function (AutoML logic)
    result_data = train_model(file_path, target_column, problem_type)

    # Format the training summary into HTML
    summary_html = format_training_summary(result_data)

    # Assume the result_data contains the leaderboard HTML (or a placeholder)
    leaderboard_html = result_data.get("leaderboard", "<p>No leaderboard available</p>")

    # Render the result page with the leaderboard and summary
    return render_template('result.html', results=leaderboard_html, summary=summary_html)

# Define the route for handling CSV file uploads
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    # Set up the folder to store uploaded files
    uploads_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(uploads_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Get the uploaded file from the request
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400  # Return error response if no file is selected

    # Save the uploaded file to the uploads folder
    file_path = os.path.join(uploads_folder, file.filename)
    file.save(file_path)

    try:
        # Attempt to read the CSV file using pandas
        df = pd.read_csv(file_path)
        
        # Get the column names from the CSV file
        column_names = df.columns.tolist()
        
        # Return the column names and file path as JSON
        return jsonify({'columns': column_names, 'file_path': file_path})
    except Exception as e:
        # Return error response if there is an issue with reading the CSV
        return jsonify({'error': str(e)}), 500

# Define the route for the chatbot interaction
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user's message from the incoming JSON request, strip whitespace, and convert to lowercase
    user_message = request.json.get("message").strip().lower()

    # Load the FAQ data from a JSON file (faq_data.json)
    with open("faq_data.json", "r") as f:
        faq_data = json.load(f)

    # Extract all FAQ questions and convert them to lowercase
    questions = [faq["question"].lower() for faq in faq_data["faqs"]]

    # Find the closest match to the user's message from the FAQ questions
    matches = difflib.get_close_matches(user_message, questions, n=1, cutoff=0.8)

    # If a match is found, return the corresponding FAQ answer
    if matches:
        for faq in faq_data["faqs"]:
            if faq["question"].lower() == matches[0]:
                return jsonify({"response": faq["answer"]})

    # If no match is found, fallback to Gemini (AI chatbot)
    try:
        # Send the user's message to the Gemini model and get the response stream
        response = chat.send_message(user_message, stream=True)
        full_response = ""
        
        # Collect the response chunks and concatenate them
        for chunk in response:
            full_response += chunk.text
        
        # Return the full response from Gemini as JSON
        return jsonify({"response": full_response})
    except Exception as e:
        # If there's an error with the Gemini response, return the error message as JSON
        return jsonify({"response": f"Error: {str(e)}"})

# Run the Flask app with debug mode enabled (auto-reload and detailed error messages)
if __name__ == '__main__':
    app.run(debug=True)
