# 😊 Facial Emotion Recognition System

A Convolutional Neural Network (CNN)-based deep learning model that recognizes human facial emotions from images.  
This project uses the **FER-2013 dataset** and supports both model training and emotion prediction using Keras and OpenCV.

---

## 📘 Overview
This project classifies human emotions into **seven categories**:
> `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`

It performs:
- Dataset extraction and preprocessing  
- CNN model training with augmentation  
- Model optimization using callbacks  
- Emotion prediction and visualization  

---

## 🧩 Dataset Setup (Important)

### Step 1 – Download Dataset  
Download the **FER-2013 dataset** from Kaggle:  
👉 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

### Step 2 – Rename and Upload  
Rename the downloaded file as:
archive.zip

css
Copy code

Then place or upload it in the same directory as your code file (`emotion_recognition_cnn.py`).  
If using Google Colab, upload it directly to `/content/`.

### Step 3 – Extraction (Handled Automatically)
You **do not need to manually unzip** the file — the code already does it:

```python
zip_path = '/content/archive.zip'
extract_dir = '/content/fer_dataset/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
After extraction, you will have this structure:

bash
Copy code
/content/fer_dataset/
    ├── train/
    │   ├── angry/
    │   ├── happy/
    │   ├── ...
    └── test/
        ├── angry/
        ├── happy/
        ├── ...
⚠️ Make sure the extracted folder contains train and test directories with subfolders for each emotion label.
Otherwise, the code won’t find the images.

⚙️ Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition
2. Install Required Libraries
bash
Copy code
pip install tensorflow keras numpy opencv-python matplotlib
🚀 Training the Model
Run:

bash
Copy code
python emotion_recognition_cnn.py
The model will:

Train using CNN architecture

Display accuracy/loss graphs

Save the model as emotion_recognition_model.h5

🧠 Model Summary
3 convolutional blocks with Batch Normalization & Dropout

Dense layers for classification

Softmax output for 7 emotion classes

📊 Example Output
csharp
Copy code
✅ Test Accuracy: 56.24%
Model saved as emotion_recognition_model.h5
🖼️ Predict Emotion from Image
To test with an image:

python
Copy code
image_path = "/content/13.jpeg"
pred_label, conf = predict_emotion(image_path)
Output Example:

yaml
Copy code
Predicted Emotion: Happy, Confidence: 92.35%
🧰 Technologies Used
Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib
