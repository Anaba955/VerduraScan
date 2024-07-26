import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import requests

# Load the model
model_path = 'trained_model.h5'  # Update this with the correct path to your model
model = load_model(model_path)

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    disease_class = np.argmax(predictions[0], axis=1)
    severity_class = np.argmax(predictions[1], axis=1)
    return disease_class[0], severity_class[0]

def get_weather_data(city_name, api_key):
    base_url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
    response = requests.get(base_url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        return None

def get_recommendation(disease_class, severity_class, weather_data):
    temp_celsius = weather_data['main']['temp'] - 273.15  # Convert Kelvin to Celsius
    recommendations = {
        0: {  # Example for Disease 0
            0: "Mild treatment for disease 0. Water regularly.",
            1: "Moderate treatment for disease 0. Apply fungicide.",
            2: "Severe treatment for disease 0. Remove infected plants."
        },
        1: {  # Example for Disease 1
            0: "Mild treatment for disease 1. Prune affected areas.",
            1: "Moderate treatment for disease 1. Use insecticide.",
            2: "Severe treatment for disease 1. Burn infected plants."
        }
        # Add more diseases and severity levels
    }

    if disease_class in recommendations and severity_class in recommendations[disease_class]:
        recommendation = recommendations[disease_class][severity_class]
        if temp_celsius > 30:
            recommendation += " Be cautious of high temperatures."
        else:
            recommendation += " Best to keep the plant in a shaded area."
        return recommendation
    return "No specific recommendation available."

# Real-time disease detection with webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    disease_class, severity_class = predict_image(frame)
    weather_data = get_weather_data('Hyderabad', 'your_openweathermap_api_key')  # Replace with actual API key

    if weather_data:
        recommendation = get_recommendation(disease_class, severity_class, weather_data)
    else:
        recommendation = "Weather data not available."

    cv2.putText(frame, f'Disease: {disease_class}, Severity: {severity_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Recommendation: {recommendation}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-time Disease Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()
