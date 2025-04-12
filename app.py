import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.layers import Multiply, BatchNormalization
from tensorflow.keras.utils import custom_object_scope

st.title("EfficientNetB1 Model for Cancer Detection using Biopsy Images")
st.write("Upload an image to detect cancer and visualize the heatmap.")
st.sidebar.write(f"TensorFlow Version: {tf.__version__}")

# Load model safely without TFOpLambda or Lambda layers
@st.cache_resource
def load_model():
    try:
        model_path = 'efficientnetb1_modelv2.h5'

        # Define known custom layers (excluding Lambda/TFOpLambda)
        custom_objects = {
            'Multiply': Multiply,
            'BatchNormalization': BatchNormalization,
        }

        with custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(model_path, compile=False)

        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

loaded_model = load_model()

# Prediction and heatmap function
def predict_image_with_heatmap(img):
    if loaded_model is None:
        return None, "Model not loaded"
    
    # Preprocess image
    img_copy = img.copy()
    img_3d = cv2.resize(img_copy, (256, 256))
    img_3d = np.array(img_3d).reshape(-1, 256, 256, 3).astype("float32") / 255.0
    
    # Predict
    prediction = loaded_model.predict(img_3d)[0]
    class_labels = ["No Cancer", "Adenocarcinoma", "Squamous Cell Carcinoma"]
    predicted_class_index = np.argmax(prediction)
    
    if predicted_class_index < len(class_labels):
        predicted_class = class_labels[predicted_class_index]
        confidence_percentage = prediction[predicted_class_index] * 100
        
        if confidence_percentage > 99.96:
            # Grad-CAM Heatmap
            last_conv_layer = loaded_model.get_layer("top_activation")
            grad_model = tf.keras.models.Model(
                [loaded_model.input], [last_conv_layer.output, loaded_model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_3d)
                loss = predictions[:, predicted_class_index]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
            heatmap = heatmap.numpy()
            
            # Resize and color the heatmap
            heatmap = cv2.resize(heatmap, (img_copy.shape[1], img_copy.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(heatmap, 0.5, img_copy, 0.5, 0)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            
            return overlay_rgb, f"Predicted: {predicted_class} (Confidence: {confidence_percentage:.2f}%)"
        else:
            return None, "Wrong Image (Confidence too low)"
    else:
        return None, "Prediction Error"

# Upload and analyze image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    st.subheader("Uploaded Image")
    st.image(img, caption='Original Image', use_column_width=True)
    
    with st.spinner("Analyzing the image..."):
        heatmap, label = predict_image_with_heatmap(img)
    
    if heatmap is not None:
        st.subheader("Heatmap Visualization")
        st.image(heatmap, caption='Heatmap Overlay', use_column_width=True)
    
    st.subheader("Prediction Result")
    st.write(label)
else:
    st.info("Please upload an image to get started.")
