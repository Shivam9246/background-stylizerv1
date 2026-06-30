import streamlit as st
import numpy as np
import tensorflow_hub as hub
from functions import load_and_preprocess_image
from PIL import Image
from ultralytics import YOLO
import io
import cv2


# ✅ Cache models so they don't reload on every run
@st.cache_resource
def load_hub_module():
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    return hub.load(hub_handle)

@st.cache_resource
def load_yolo():
    return YOLO("yolov8m-seg.pt")


# --- UI ---
st.markdown("<h1 style='text-align: center;'>🎨 Background Stylizer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #1E90FF;'>Edit the Styles of the Background of Your Images</h3>", unsafe_allow_html=True)
st.markdown("---")

st.info("⏳ On first load, models may take a minute to download. Please be patient.")

with st.form(key='image_upload_form'):
    st.markdown("<h2 style='font-weight:bold;'>📷 Upload Content Image</h2>", unsafe_allow_html=True)
    st.caption("An image containing a person whose background you want to stylize")
    contentimg = st.file_uploader("", type=["jpg", "jpeg", "png"], key='image1')

    st.markdown("<h2 style='font-weight:bold;'>🖼️ Upload Style Image</h2>", unsafe_allow_html=True)
    st.caption("An artwork or texture to apply to the background")
    styleimg = st.file_uploader("", type=["jpg", "jpeg", "png"], key='image2')

    submit_button = st.form_submit_button(label='✨ Stylize Background')

if submit_button:
    if contentimg is not None and styleimg is not None:

        content = Image.open(contentimg).convert("RGB")
        style = Image.open(styleimg).convert("RGB")

        with st.spinner("Loading models..."):
            hub_module = load_hub_module()
            yolo_model = load_yolo()

        # --- Step 1: Neural Style Transfer ---
        with st.spinner("Applying style transfer to background..."):
            content_img = load_and_preprocess_image(image=content)
            style_img = load_and_preprocess_image(image=style)

            outputs = hub_module(content_img, style_img)
            stylized_tensor = outputs[0]

            # Convert tensor to PIL image (in memory — no disk write)
            stylized_np = (stylized_tensor.numpy()[0] * 255).astype(np.uint8)
            background = Image.fromarray(stylized_np).resize((256, 256)).convert('RGBA')

        # --- Step 2: YOLO Segmentation ---
        with st.spinner("Detecting and segmenting person..."):
            resized_image = content.resize((256, 256))
            results = yolo_model.predict(resized_image, verbose=False)
            result = results[0]

            # ✅ Guard: no person detected
            if result.masks is None or len(result.masks) == 0:
                st.error("❌ No person detected in the content image. Please upload a clear image of a person.")
            else:
                mask1 = result.masks[0]
                polygon = mask1.xy[0]

                resized_array = np.array(resized_image)  # PIL → numpy (already RGB)
                masky = np.zeros_like(resized_array[:, :, 0])
                cv2.fillPoly(masky, [polygon.astype(np.int32)], 255)

                # Extract person pixels (no BGR→RGB flip needed, PIL is already RGB)
                overlay = cv2.bitwise_and(resized_array, resized_array, mask=masky)
                r, g, b = cv2.split(overlay)
                overlay_rgba = cv2.merge((r, g, b, masky))
                overlay_image = Image.fromarray(overlay_rgba, 'RGBA')

                # --- Step 3: Composite person onto styled background ---
                combined = Image.alpha_composite(background, overlay_image)

                st.markdown("---")
                st.markdown("### ✅ Result")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(content.resize((256, 256)), caption="Original")
                with col2:
                    st.image(Image.fromarray(stylized_np).resize((256, 256)), caption="Styled Background")
                with col3:
                    st.image(combined, caption="Final Output")

                # Download button
                img_byte_arr = io.BytesIO()
                combined.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                st.download_button(
                    label="⬇️ Download Stylized Image",
                    data=img_byte_arr,
                    file_name="stylized_image.png",
                    mime="image/png"
                )
    else:
        st.error("⚠️ Please upload both a content image and a style image.")
