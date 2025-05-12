# Thyroid Detection and Classification Using DNN Based on Hybrid Meta-Heuristic and LSTM Technique

## 🔍 Overview

This project presents an AI-powered system for the detection and classification of thyroid disorders using ultrasound imaging. The system is built with Deep Neural Networks (DNN), Long Short-Term Memory (LSTM), and a hybrid meta-heuristic algorithm combining Grey Wolf Optimizer (GWO) and Particle Swarm Optimization (PSO) for feature selection. The frontend is developed with Streamlit, allowing users to easily upload thyroid scans and receive real-time diagnostic results.

---

## 🧠 Key Features

* **Ultrasound Image Classification**
* **Hybrid DNN + LSTM Architecture**
* **Feature Optimization with GWO-PSO**
* **Real-time Image Segmentation using YOLOv5**
* **Performance Metrics (Accuracy, SSIM, PSNR, AUC, etc.)**
* **Streamlit Web Interface with Audio Feedback**
* **Segmentation Visualization and Classification Feedback with Recommendations**

---

## 🗂️ Project Structure

```
├── app.py                  # Streamlit-based user interface
├── proposed.py            # Model training and evaluation logic
├── model.h5               # Pre-trained deep learning model
├── Dataset/               # Labeled ultrasound images
├── Background/            # Background images for the app UI
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

---

## 💻 System Requirements

### Hardware:

* **Processor:** Intel Core i5/i7 or AMD Ryzen 5/7
* **GPU:** NVIDIA RTX with CUDA support (recommended for training)
* **RAM:** 8 GB (minimum), 16 GB (recommended)
* **Storage:** 512 GB SSD or higher
* **Operating System:** Windows 10/11

### Software:

* **Language:** Python 3.7+
* **IDE:** Spyder / VSCode
* **Libraries & Frameworks:**

  * TensorFlow / Keras
  * OpenCV
  * NumPy, Pandas, Matplotlib
  * Streamlit
  * gTTS
  * Scikit-learn

---

## 📈 Model Details

* **Image Size:** 65x65 pixels
* **Models Used:** CNN, DenseNet121, VGG16, VGG19, LSTM
* **Ensemble:** Weighted average of DNN and LSTM outputs
* **Optimization:** PCA + GWO-PSO hybrid feature selection
* **Metrics:** Accuracy, Precision, Recall, F1-score, ROC, PSNR, SSIM

---

## 🧩 Additional Highlights

* **Audio Output:** Disease predictions are also played back using text-to-speech (gTTS)
* **Segmented Output Display:** Visuals showing identified abnormal thyroid regions
* **Recommendation System:** Text-based advice based on classification result
* **Voice-assisted Feedback:** Enhances user accessibility and experience

---

## 🚀 Future Enhancements

* Transfer Learning from medical pre-trained networks
* Federated Learning for privacy-compliant training
* Integration of blood test reports and voice input
* EHR integration and cloud deployment

---

## 📜 License

This project is a final year academic submission under K.Ramakrishnan College of Technology. Use is permitted for academic and non-commercial research only.

---

## 👩‍💻 Authors

* ABDUR RAZIQ FAREED R (811721243002)
* ABILASH K (811721243003)
* ISHAN SANJEEV R (811721243018)
* MOHAMMED ARSHAD ROSHAN A (8117212433032)
