
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Photo Lab", page_icon="📸", layout="wide")

st.title("📸 Photo Lab")
st.write("Upload a photo and experiment with brightness, contrast, ""and thresholding using OpenCV.")

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image and convert it to OpenCV's BGR format
    pil_image = Image.open(uploaded_file).convert("RGB")
    original = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.sidebar.header("Editing tools")

    brightness = st.sidebar.slider("Brightness", -100, 100, 0 )
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, step=0.1)
    clipping = st.sidebar.checkbox("Clip pixel values", value=True)

    threshold_method = st.sidebar.selectbox("Thresholding",[
            "None",
            "Global",
            "Inverse",
            "Adaptive",
            "Otsu"])
    threshold_value = st.sidebar.slider("Threshold value", 0, 255, 127)

    # 1. Brightness: use cv2.add() or cv2.subtract()
    if brightness > 0:
        bright_image = cv2.add(original, np.full(original.shape, brightness, dtype=np.uint8))
    elif brightness < 0:
        bright_image = cv2.subtract(original, np.full(original.shape, abs(brightness), dtype=np.uint8))
    else:
        bright_image = original.copy()

    # 2. Contrast: use cv2.multiply()
    float_image = bright_image.astype(np.float32)
    contrast_image = cv2.multiply(float_image, contrast)

    if clipping:
        contrast_image = np.clip(contrast_image, 0, 255)

    edited = contrast_image.astype(np.uint8)

    # 3. Thresholding: create a black-and-white version
    if threshold_method != "None":
        gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)

        if threshold_method == "Global":
            _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_method == "Inverse":
            _, result = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        elif threshold_method == "Adaptive":
            # Adaptive thresholding requires an odd block size.
            result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:  # Otsu
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        processed = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        processed = edited

    # Display original and processed images side by side
    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)

    with right:
        st.subheader("Processed image")
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

    # 4. Let the user download the edited image
    output_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_rgb)
    download_buffer = BytesIO()
    output_image.save(download_buffer, format="PNG")

    st.download_button(label="Download edited image", data=download_buffer.getvalue(), file_name="photo_lab_edited.png", mime="image/png")
else:
    st.info("Upload a photo above to start editing.")
