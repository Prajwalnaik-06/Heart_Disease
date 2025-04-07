from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('heart_model.h5')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    features = [
        float(request.form['age']),
        float(request.form['sex']),
        float(request.form['cp']),
        float(request.form['trestbps']),
        float(request.form['chol']),
        float(request.form['fbs']),
        float(request.form['restecg']),
        float(request.form['thalach']),
        float(request.form['exang']),
        float(request.form['oldpeak']),
        float(request.form['slope']),
        float(request.form['ca']),
        float(request.form['thal'])
    ]
    
    # Convert features to numpy array
    input_features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)
    
    # Generate a plot for visualization
    plt.figure()
    labels = ['No Heart Disease', 'Heart Disease']
    probabilities = [1 - prediction[0][0], prediction[0][0]]
    
    plt.bar(labels, probabilities, color=['green', 'red'])
    plt.title('Heart Disease Prediction')
    plt.ylabel('Probability')
    plt.ylim(0, 1)  # Set y-axis limits to 0-1
    plt.axhline(y=0.5, color='gray', linestyle='--')  # Add a horizontal line at y=0.5
    plt.text(0, probabilities[0] + 0.02, f'{probabilities[0]:.2f}', ha='center', color='black')
    plt.text(1, probabilities[1] + 0.02, f'{probabilities[1]:.2f}', ha='center', color='black')

    # Add a clear message indicating the prediction
    prediction_message = "Heart Disease" if prediction[0][0] > 0.5 else "No Heart Disease"
    plt.text(0.5, 0.5, prediction_message, fontsize=15, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    
    # Save the plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)  # Move to the beginning of the BytesIO object
    plt.close()  # Close the plot to free memory

    # Encode the image to base64
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Return the prediction and the plot
    return render_template('result.html', prediction=int(prediction[0][0]), plot=img_base64)

if __name__ == '__main__':
    app.run(debug=True)