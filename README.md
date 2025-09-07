cat > README.md << EOF
# Sign Language Detector 🖐️🤟

A **real-time sign language detection system** built with **Python, OpenCV, and Mediapipe**.
This project enables gesture recognition via webcam, model training, and deployment through a simple web interface.

---

## 🚀 Features

* 📷 **Image Collection** – Capture training images from your webcam.
* 📂 **Dataset Builder** – Create structured datasets for training.
* 🧠 **Model Training** – Train a custom classifier on collected data.
* 🎯 **Real-time Inference** – Detect signs/gestures live through webcam.
* 📊 **Metrics & Explainability** – Evaluate with accuracy, confusion matrix, LIME & SHAP.
* 🌐 **Web Interface** – Run detection inside your browser.
* 🔊 **Speech Output (optional)** – Convert recognized signs into speech.

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Tejascodex001/sign.git
cd sign
```

2. **Create & activate a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1. Collect Images

Capture gesture images using webcam:

```bash
python collect_imgs.py
```

⚠️ Keep camera distance consistent for all samples.

---

### 2. Create Dataset

Organize images into a dataset for training:

```bash
python create_dataset.py
```

---

### 3. Train Classifier

Train a sign language classifier:

```bash
python train_classifier.py
```

---

### 4. Run Inference

Test your model in real time:

```bash
python inference_classifier.py
```

---

### 5. Launch Web Interface

Start server and open \`http://localhost:5000\`:

```bash
python server.py
```

---

## 📊 Results

The project generates:

* `accuracy_score.png` – Accuracy visualization
* `classification_report.png` – Precision, Recall, F1
* `confusion_matrix.png` – Class-wise performance
* `lime_explanation.png`, `shap_summary_plot.png` – Model interpretability

---

## 📂 Repository Structure

```bash
sign/
│── collect_imgs.py          # Capture images via webcam
│── create_dataset.py        # Build dataset
│── train_classifier.py      # Train ML classifier
│── inference_classifier.py  # Run live detection
│── server.py                # Web interface server
│── speech.py                # Convert predictions to speech
│── explain.py               # Model interpretability
│── generate_metrics.py      # Accuracy, confusion matrix, etc.
│── index.html               # Web UI
│── requirements.txt         # Dependencies
│── data.pickle              # Stored dataset
│── detected_letters.txt     # Predictions log
│── *.png                    # Metrics visualizations
```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch (\`feature/new-sign\`)
3. Commit changes
4. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this project.

---
EOF
