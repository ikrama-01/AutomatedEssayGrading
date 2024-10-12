from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow import keras
import os

app = Flask(__name__)

# Load the saved model
model = keras.models.load_model('essay_grading_model.keras')

# Check if the vectorizer file exists
if os.path.exists('vectorizer.pkl'):
    print("Vectorizer file found")
    try:
        # Load the vectorizer
        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
            print("Vectorizer loaded successfully.")
            print(f"IDF values: {vectorizer.idf_}") 
    except EOFError:
        print("Error loading vectorizer: Ran out of input. The file may be corrupted.")
        exit(1)  # Exit the application if the vectorizer can't be loaded
else:
    print("Vectorizer file not found")
    exit(1)  # Exit the application if the vectorizer file is missing

# Define the home route to display the file upload form
@app.route('/')
def home():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Essay Grading Application</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
            }
            h2 {
                color: #333;
                text-align: center;
            }
            .container {
                max-width: 600px;
                margin: auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            input[type="file"] {
                display: block;
                margin: 20px auto;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                width: 100%;
            }
            input[type="submit"] {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                cursor: pointer;
                display: block;
                margin: auto;
            }
            input[type="submit"]:hover {
                background-color: #218838;
            }
            .result {
                text-align: center;
                margin-top: 20px;
                font-size: 1.5em;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Essay Grading Application</h2>
            <form action="/grade" method="post" enctype="multipart/form-data">
                <label for="file">Upload your essay (txt file):</label>
                <input type="file" name="file" accept=".txt" required>
                <input type="submit" value="Upload">
            </form>
        </div>
    </body>
    </html>
    '''

# Define the grading route that handles file uploads and returns the score
@app.route('/grade', methods=['POST'])
def grade_essay():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No selected file", 400

    # Read the uploaded file and decode it to a string
    essay_content = file.read().decode('utf-8')
    

    # Process the essay with the vectorizer
    try:
        essay_vector = vectorizer.transform([essay_content]).toarray()
    except Exception as e:
        return f"Error processing essay: {str(e)}", 500

    # Use the model to predict the grade
    predicted_score = model.predict(essay_vector)


    # Example structure for calculating and displaying MAE
    predicted_score = model.predict(essay_vector)
    true_score = ...  # This would normally be fetched or known
    mae = np.mean(np.abs(predicted_score - true_score))
    print(f"Mean Absolute Error: {mae:.2f}")

    return f'''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Essay Grading Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 600px;
                    margin: auto;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }}
                .result {{
                    font-size: 2em;
                    color: #333;
                    margin: 20px 0;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }}
                a:hover {{
                    background-color: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Essay Grading Result</h2>
                <div class="result">Predicted Score: {predicted_score:.2f}</div>
                <a href="/">Upload Another Essay</a>
            </div>
        </body>
        </html>
        '''


if __name__ == '__main__':
    app.run(debug=True)
