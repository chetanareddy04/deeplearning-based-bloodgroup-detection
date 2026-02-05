# 🩸  Deep Learning–Based Blood Group Detection Using Fingerprint and Blood Images


A deep learning–based web application that predicts human blood groups using blood sample images and fingerprint images through MobileNetV2 and Flask deployment.

---
## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Tools & Technologies](#-tools--technologies)
- [Methodology](#-methodology)
- [Key Insights](#-key-insights)
- [Model Output](#-model-output)
- [How to Run This Project](#-how-to-run-this-project)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Conclusion](#-conclusion)
- [Future Work](#-future-work)
- [Author](#-author)
- [Contact](#-contact)

## 🔎 Overview
This project focuses on applying deep learning techniques to automate blood group detection using image-based data. Two independent approaches are explored: one using blood sample images and another using fingerprint images. Convolutional Neural Networks (CNNs) are trained to learn visual patterns from these images and classify them into corresponding blood groups. The trained models are integrated into a Flask-based web application that allows users to upload images and receive predictions in real time.

---

## 📖 Problem Statement
Accurate blood group identification is critical in medical emergencies, blood transfusions, and healthcare systems. Traditional blood typing methods require laboratory equipment, trained personnel, and time. This project explores an alternative approach using deep learning and image-based analysis to automate blood group detection from blood images and fingerprint images.

---

## 📊 Dataset
- Blood image dataset collected from publicly available sources (Kaggle)
- Fingerprint image dataset used to analyze biometric patterns
- Images are labeled with blood groups: A+, A−, B+, B−, AB+, AB−, O+, O−

> Note: Due to large dataset size, the datasets are not included in this repository.

---

## 🛠️ Tools & Technologies
- Programming Language: Python
- Deep Learning Framework: TensorFlow, Keras
- Model Architecture: MobilenetV2 (CNN)
- Web Framework: Flask
- Image Processing: OpenCV
- Libraries: NumPy, Matplotlib
- Frontend: HTML, CSS,JavaScript

---

## ⚙️ Methodology

The project follows a deep learning–based image classification pipeline to predict blood groups using both blood sample images and fingerprint images.

### Step 1: Data Collection
- Blood group image dataset collected from Kaggle
- Fingerprint image dataset used for non-invasive prediction
- Images organized into folders based on blood group classes

### Step 2: Data Preprocessing
- Image resizing to **224 × 224**
- Pixel value normalization
- Conversion of images into arrays
- Dataset splitting into training and testing sets

### Step 3: Model Development
- Transfer learning using **MobileNetV2**
- CNN-based classification model
- Final dense layer added for **8 blood group classes**
  - A+, A-, B+, B-, AB+, AB-, O+, O-

### Step 4: Model Training
- Models trained separately for:
  - Blood image dataset
  - Fingerprint image dataset
- Optimization using Adam optimizer
- Loss function: categorical crossentropy

### Step 5: Model Evaluation
- Performance measured using:
  - Accuracy
  - Confusion Matrix
  - Performance analysis charts

### Step 6: Deployment
- Models saved as **.h5 files**
- Flask web application created for prediction
- Users can upload an image and get real-time prediction results

---

## 💡 Key Insights
- Blood image–based detection showed more stable performance due to clearer visual features.
- Fingerprint-based detection produced promising results but depended on image quality.
- Deep learning proved effective for medical and biometric image classification tasks.

---

## 🖼️ Model Output
- Predicted blood group label
- Uploaded image preview
- Performance metrics visualized through evaluation charts

Screenshots of UI, predictions, and performance analysis are available in the `screenshots/` folder.

---

## ▶️ How to Run This Project

1. Clone the repository:
   ```bash
   git clone <repository-url>
2. Navigate to the project directory:
   ```bash
   cd <project-folder-name>
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the flask application:
   ```bash
   python app.py
5. Open the application in your browser:
   ```cpp
   http://127.0.0.1:5000/
6. Upload an image:

   * Choose either a blood image or fingerprint image

   * Click on predict to view the blood group result

## 🗂️ Project Structure

    deep-learning-based-blood-group-detection/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    │
    ├── static/
    │   └── test   
    │       ├── sample blood.jpg
    │       └── sample fingerprint.BPM
    │
    └── screenshots/
        ├── blood input.png
        ├── blood prediction.png
        ├── blood performance.png
        ├── fingerprint input.png
        ├── fingerprint prediction.png
        └── fingerprint performance.png


   
## 📈 Results

The deep learning models successfully classified blood groups from both blood sample images and fingerprint images. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix demonstrated effective learning and reliable prediction performance.

The blood image–based approach showed more consistent and stable results due to clearer visual patterns, while the fingerprint-based approach produced promising results when high-quality images were provided.

> **Note:** All outputs of the project are saved in the `screenshots` folder, organized as follows:  
> 
> **Blood images:**  
> - `blood input.png` – input image provided to the model  
> - `blood prediction.png` – model's predicted output 
> - `blood performance.png` – performance evaluation   
> 
> **Fingerprint images:**  
> - `fingerprint input.png` – input image provided to the model  
> - `fingerprint prediction.png` – model's predicted output  
> - `fingerprint performance.png` – performance evaluation 


## ✅ Conclusion

This project demonstrates the effective use of deep learning techniques for blood group detection using both blood images and fingerprint images. By leveraging the MobileNetV2 architecture and deploying the model through a Flask-based web application, the system delivers reliable and real-time predictions.

The inclusion of a non-invasive fingerprint-based approach alongside traditional blood image analysis highlights the potential of AI-driven solutions to improve efficiency and accessibility in medical diagnostics, particularly in clinical and remote healthcare environments.

## 🚀 Future Work

Future enhancements can further improve the system’s accuracy and real-world applicability, including:
- Expanding detection using additional non-invasive biometric features
- Developing a mobile application for field and remote usage
- Experimenting with advanced deep learning architectures such as EfficientNet or Vision Transformers
- Training on larger and more diverse datasets for better generalization
- Integrating real-time image capture and multi-platform support
- Conducting clinical validation and regulatory approval studies

## 👤 Author

**Sama Chetana**  
B.Tech Graduate in Artificial Intelligence & Data Science  
Aspiring AI / Data Science / Machine Learning Engineer  

---

## 📬 Contact

- 📧 Email: chetanareddys71@gmail.com 
- 💼 LinkedIn: [https://www.linkedin.com/in/sama-chetana-71142031b]
- 🐙 GitHub: [https://github.com/chetanareddy04]


