# 🎨 Background Stylizer

A Streamlit web app that stylizes the **background** of an image while keeping the **person** intact — using Neural Style Transfer and YOLO semantic segmentation.

## How It Works

1. **Upload a content image** — a photo with a person in it
2. **Upload a style image** — any artwork or texture
3. The app:
   - Applies **Neural Style Transfer** (via TensorFlow Hub) to the full image
   - Uses **YOLOv8 segmentation** to detect and extract the person
   - Composites the person back onto the styled background

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- pip

### Install & Run
```bash
git clone https://github.com/YOUR_USERNAME/background-stylizer.git
cd background-stylizer
pip install -r requirements.txt
streamlit run main.py
```

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `main.py`
5. Click **Deploy**

## 📁 File Structure

```
background-stylizer/
├── main.py            # Streamlit UI + app logic
├── functions.py       # Image preprocessing helpers
├── requirements.txt   # Python dependencies
├── packages.txt       # System packages for Streamlit Cloud
└── README.md
```

## References
- [TF Hub: Arbitrary Image Stylization](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization)
- [YOLOv8 Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
