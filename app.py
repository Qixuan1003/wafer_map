import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import io
import zipfile
import os
import gdown
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Wafer Defect Classifier",
    page_icon="🔍",
    layout="wide"
)

# Define the class mappings
CLASS_MAPPING = {
    'C1': 'Normal',
    'C2': 'Center (C)',
    'C3': 'Donut (D)',
    'C4': 'Edge_Loc (EL)',
    'C5': 'Edge_Ring (ER)',
    'C6': 'Loc (L)',
    'C7': 'Near_Full (NF)',
    'C8': 'Scratch (S)',
    'C9': 'Random (R)',
    'C10': 'C+EL',
    'C11': 'C+ER',
    'C12': 'C+L',
    'C13': 'C+S',
    'C14': 'D+EL',
    'C15': 'D+ER',
    'C16': 'D+L',
    'C17': 'D+S',
    'C18': 'EL+L',
    'C19': 'EL+S',
    'C20': 'ER+L',
    'C21': 'ER+S',
    'C22': 'L+S',
    'C23': 'C+EL+L',
    'C24': 'C+EL+S',
    'C25': 'C+ER+L',
    'C26': 'C+ER+S',
    'C27': 'C+L+S',
    'C28': 'D+EL+L',
    'C29': 'D+EL+S',
    'C30': 'D+ER+L',
    'C31': 'D+ER+S',
    'C32': 'D+L+S',
    'C33': 'EL+L+S',
    'C34': 'ER+L+S',
    'C35': 'C+L+EL+S',
    'C36': 'C+L+ER+S',
    'C37': 'D+L+EL+S',
    'C38': 'D+L+ER+S'
}

# Add a navigation menu in the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Single Image Classifier", "Batch Processing", "Reference Guide", "Model Information"])

@st.cache_resource
def load_model():
    """Load and cache the model from Google Drive"""
    try:
        # Google Drive file ID
        file_id = "1Gv0tu4t1rfV2BrTFRqP0p50UZuFDq8Pz"
        output_path = "mobilenetv2_none.keras"
        
        # Download file if it doesn't exist
        if not os.path.exists(output_path):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            
            if not os.path.exists(output_path):
                st.error("Failed to download model file")
                return None
        
        # Load the model
        model = tf.keras.models.load_model(output_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model inference"""
    if isinstance(image, Image.Image):
        # Convert image to grayscale since wafer maps are binary/grayscale
        image = image.convert('L')
        image = np.array(image)

    # Normalize pixel values to [0,1] range
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0

    # Add RGB channels
    image = np.stack([image] * 3, axis=-1)

    # Resize to model input size
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, 0)
    return image

def predict_with_timing(model, image):
    """Make prediction with timing information"""
    start_time = time.time()
    processed_image = preprocess_image(image)
    
    predictions = model.predict(processed_image, verbose=0)
    inference_time = (time.time() - start_time) * 1000  # ms

    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    results = []
    for idx in top_3_idx:
        class_name = f'C{idx+1}'
        defect_type = CLASS_MAPPING[class_name]
        probability = float(predictions[0][idx])
        results.append((class_name, defect_type, probability))

    return results, inference_time

def process_batch_images(model, images):
    """Process multiple images and return results"""
    results = []
    total_time = 0
    
    for img_name, img in images:
        try:
            predictions, inference_time = predict_with_timing(model, img)
            top_prediction = predictions[0]  # Get the highest confidence prediction
            results.append({
                'Image': img_name,
                'Predicted Class': f"{top_prediction[0]} ({top_prediction[1]})",
                'Confidence': f"{top_prediction[2]*100:.1f}%",
                'Inference Time (ms)': f"{inference_time:.1f}"
            })
            total_time += inference_time
        except Exception as e:
            results.append({
                'Image': img_name,
                'Predicted Class': f'Error: {str(e)}',
                'Confidence': 'N/A',
                'Inference Time (ms)': 'N/A'
            })
            
    return results, total_time

# Load the model at startup
model = load_model()

# Main application logic based on selected page
if page == "Single Image Classifier":
    st.title("🔍 Wafer Defect Pattern Classification")
    
    # File uploader for single image
    st.write("### Upload a wafer map image for classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            # Display original image and predictions side by side
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            if model is not None:
                # Make prediction
                results, inference_time = predict_with_timing(model, image)

                with col2:
                    st.write("#### Prediction Results")
                    st.markdown(f"**Inference Time:** {inference_time:.1f} ms")

                    # Display predictions
                    for class_name, defect_type, probability in results:
                        confidence = probability * 100
                        st.markdown(f"""
                        <div style='padding: 10px; margin: 5px 0; border-radius: 5px;
                        background-color: rgba(0, 120, 200, 0.1);'>
                        <strong>{class_name}</strong> ({defect_type})<br/>
                        Confidence: {confidence:.1f}%
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

elif page == "Batch Processing":
    st.title("📦 Batch Image Processing")
    
    st.write("""
    ### Upload multiple wafer map images for batch processing
    You can upload multiple images or a ZIP file containing images.
    """)
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader("Choose images or ZIP file", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)
    
    if uploaded_files:
        images_to_process = []
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name.endswith('.zip'):
                    # Handle ZIP file
                    with zipfile.ZipFile(uploaded_file) as z:
                        for filename in z.namelist():
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                with z.open(filename) as f:
                                    img_data = io.BytesIO(f.read())
                                    images_to_process.append((filename, Image.open(img_data)))
                else:
                    # Handle individual image files
                    images_to_process.append((uploaded_file.name, Image.open(uploaded_file)))
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if images_to_process:
            st.write(f"Processing {len(images_to_process)} images...")
            
            if model is not None:
                results, total_time = process_batch_images(model, images_to_process)
                
                # Display results
                st.write("### Results")
                st.write(f"Total processing time: {total_time:.1f} ms")
                
                # Create a DataFrame for better display
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Option to download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="batch_processing_results.csv",
                    mime="text/csv"
                )

elif page == "Reference Guide":
    st.title("📚 Wafer Defect Patterns Reference Guide")

    st.write("""
    ### Understanding Wafer Defect Patterns
    This guide shows examples of all 38 defect pattern classes. The patterns are grouped into four categories:
    - Single type patterns (C1-C9)
    - 2 Mixed-type patterns (C10-C22)
    - 3 Mixed-type patterns (C23-C34)
    - 4 Mixed-type patterns (C35-C38)
    """)

    # Add pattern descriptions
    st.write("### Pattern Descriptions")

    # Group patterns by type
    single_type = {k: v for k, v in CLASS_MAPPING.items() if k in [f'C{i}' for i in range(1, 10)]}
    two_mixed = {k: v for k, v in CLASS_MAPPING.items() if k in [f'C{i}' for i in range(10, 23)]}
    three_mixed = {k: v for k, v in CLASS_MAPPING.items() if k in [f'C{i}' for i in range(23, 35)]}
    four_mixed = {k: v for k, v in CLASS_MAPPING.items() if k in [f'C{i}' for i in range(35, 39)]}

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Single Type Patterns")
        for k, v in single_type.items():
            st.write(f"**{k}**: {v}")

        st.write("#### 2 Mixed-Type Patterns")
        for k, v in two_mixed.items():
            st.write(f"**{k}**: {v}")

    with col2:
        st.write("#### 3 Mixed-Type Patterns")
        for k, v in three_mixed.items():
            st.write(f"**{k}**: {v}")

        st.write("#### 4 Mixed-Type Patterns")
        for k, v in four_mixed.items():
            st.write(f"**{k}**: {v}")
            
    st.write("### Sample Images for All Defect Types")
    
    # Use the raw content URL from GitHub
    image_url = "https://raw.githubusercontent.com/Qixuan1003/wafer_map/main/all.png"
    
    try:
        st.image(image_url, caption="Wafer Defect Pattern Examples showing all 38 pattern types", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading reference image: {str(e)}")
        st.info("""
        Common wafer defect pattern characteristics:
        - **Normal (C1)**: Uniform pattern with no defects
        - **Center (C2)**: Defects concentrated in the center
        - **Donut (D)**: Ring-shaped pattern of defects
        - **Edge-Located (EL)**: Defects along one edge
        - **Edge-Ring (ER)**: Defects around the perimeter
        - **Localized (L)**: Clustered defects in specific areas
        - **Near-Full (NF)**: Most of the wafer shows defects
        - **Scratch (S)**: Linear pattern of defects
        - **Random (R)**: Randomly distributed defects
        
        Mixed patterns (C10-C38) combine characteristics of these basic patterns.
        """)
            
    st.write("### Sample Images for All Defect Types")
    
    # Define reference image URL
    image_url = "https://raw.githubusercontent.com/Qixuan1003/wafer-map/main/all.png"
    
    try:
        # Create a markdown image
        st.markdown(f"![Wafer Defect Pattern Examples]({image_url})")
        st.caption("Wafer Defect Pattern Examples showing all 38 pattern types")

elif page == "Model Information":
    st.title("📊 Model Information")

    st.write("""
    ### MobileNetV2 Model Specifications
    
    This application uses a fine-tuned MobileNetV2 model for wafer defect classification. Here are the key specifications:
    """)

    # Display model metrics
    metrics = {
        'Accuracy': '98.32%',
        'Model Size': '9.92 MB',
        'FLOPs': '614.01M',
        'Parameters': '2.60M',
        'Average Inference Time': '~1.57 ms per image'
    }

    for metric, value in metrics.items():
        st.markdown(f"""
        <div style='padding: 10px; margin: 5px 0; border-radius: 5px;
        background-color: rgba(0, 120, 200, 0.1);'>
        <strong>{metric}:</strong> {value}
        </div>
        """, unsafe_allow_html=True)

    st.write("""
    ### Model Architecture Details
    
    - Base Model: MobileNetV2 (pre-trained on ImageNet)
    - Input Size: 224x224x3
    - Output: 38 classes (various wafer defect patterns)
    - Training: Fine-tuned on wafer map dataset with balanced class distribution
    """)
