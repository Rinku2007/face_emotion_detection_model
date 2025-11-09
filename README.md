# 🧠FACE EMOTION DETECTION MODEL

## Project overview
The Face Emotion Detection system uses deep learning and computer vision to automatically recognize human emotions from facial expressions. It captures facial images through a webcam or dataset and predicts emotions such as happy, sad, angry, surprise, neutral, etc.

## Tools & Technologies Used
Programming Language:Python

Libraries & Frameworks:
    TensorFlow
    Keras
    Pandas
    NumPy
    OpenCV (opencv-contrib-python)
    scikit-learn
    tqdm
    
Development Environment: Jupyter Notebook



## Project Structure

face_emotion_detection/
│
├── dataset/                   # Training & testing images
├── trainmodel.ipynb           # Model training notebook
├── realtimedetection.py       # Real-time emotion detection script
├── haarcascade_frontalface_default.xml  # Face detection model
├── requirements.txt           # Dependencies list
├── README.md                  # Project documentation
└── .venv/                     # Virtual environment (not pushed to Git)



## Working Process 
1.Dataset Collection:
        Downloaded from Kaggle (Facial Expression Dataset).

2.Data Preprocessing:
        Images are converted to grayscale and resized to (48x48).

3.Model Building:
        A CNN (Convolutional Neural Network) model built using Keras and TensorFlow.

4.Training:
        The model is trained on labeled emotion data and validated for accuracy.

5.Real-Time Detection:
        Using OpenCV to detect faces via webcam and predict emotion in real time.


## how to run project

1. clone the repository
git clone https://github.com/<your-username>/face_emotion_detection.git
cd face_emotion_detection

2. Create and Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate      # for Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run Training Notebook
Open Jupyter Notebook and run trainmodel.ipynb.

5. Run Real-Time Detection
python realtimedetection.py



## 😊 Detected Emotions
Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral


## 📈 Results
Training Accuracy: ~71%
Model Type: CNN
Performance: Works on real-time webcam video feed

## Future Scope
Improve accuracy with deeper CNN or transfer learning (e.g., VGG16, ResNet).
Add support for multiple faces at once.
Create a web-based interface for live detection.

## Conclusion
The Face Emotion Detection project successfully demonstrates how deep learning and computer vision can identify human emotions from facial expressions in real time.
