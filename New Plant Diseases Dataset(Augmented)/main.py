import streamlit as st
import tensorflow as tf
import numpy as np
from disease_info import plant_diseases_info
import requests

# tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single  img to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence_score = np.max(prediction)  # Get the confidence score
    return result_index, confidence_score


# disease info
def get_info(prediction, score):
    st.info(f"**Confidence:** {round(score * 100, 2)}%\n\n"
        f"**Botanical name:** {plant_diseases_info[prediction]['botanical_name']}\n\n"
            f"**Common name:** {plant_diseases_info[prediction]['common_name']}\n\n"
            f"**Cause:** {plant_diseases_info[prediction]['recommendations']['cause']}\n\n"
            f"**Prevention:** \n\n"
            f"Hot weather: {plant_diseases_info[prediction]['recommendations']['prevention']['hot']}\n\n"
            f"Cold weather: {plant_diseases_info[prediction]['recommendations']['prevention']['cold']}\n\n"
            f"Rainy: {plant_diseases_info[prediction]['recommendations']['prevention']['rainy']}\n\n"
            f"**Cure:** {plant_diseases_info[prediction]['recommendations']['cure']}\n\n"
    )


# UI

# sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, width=550)
    
    st.markdown("""
    #### Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help you identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Your Plant Image:** Navigate to the **Disease Recognition** page and upload an image of your plant with suspected diseases.
    2. **Advanced Analysis:** Our system processes the image using cutting-edge algorithms to identify potential diseases.
    3. **View Results:** Get detailed insights and recommendations for further action based on the analysis.

    ### Why Choose Us?
    - **Accuracy:** Utilizing state-of-the-art machine learning techniques ensures precise disease detection.
    - **User-Friendly:** Designed for simplicity, our interface offers an intuitive experience for all users.
    - **Fast and Efficient:** Receive results swiftly, empowering quick decision-making for better crop management.

    ### Getting Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Discover more about our project and objectives on the **About** page.
    """)


# About Page
elif app_mode == "About":
    st.header("About the Dataset")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the [original dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). 

        It consists of approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. 
        The dataset is split into an 80/20 ratio for training and validation sets, preserving the directory structure. 
        An additional directory containing 33 test images was created later for prediction purposes.

        #### Dataset Structure
        - **Train:** 70,295 images used for training.
        - **Test:** 33 images reserved for testing.
        - **Validation:** 17,572 images for model validation.

        #### Purpose
        This dataset trains our ML models to accurately identify plant diseases, supporting farmers and agriculture enthusiasts in crop management.

        For more details, please refer to the [original Kaggle dataset page](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
        """)


# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # File uploader for image selection
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        # Display the chosen image
        st.image(test_image, width=400, caption="Uploaded Image")

        # Predict button
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                # Perform model prediction
                result_index, confidence_score = model_prediction(test_image)

            # Display prediction result
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"Prediction: {class_name[result_index]}")
            get_info(class_name[result_index], confidence_score)
            # st.balloons()
            # get_recommendation(class_name[result_index])

            # Optionally, display confidence scores or additional details about the prediction
            # st.info("Confidence: {:.2f}%".format(prediction_confidence * 100))
        else:
            st.warning("Please click 'Predict' to identify the disease.")

    else:
        st.info("Please upload an image to begin.")
