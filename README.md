
# 🧠 MLOps: Linear Regression Pipeline

This project is built as part of the **MLOps Major Assignment**, focused on creating a complete end-to-end machine learning pipeline using **Linear Regression** with proper CI/CD, Dockerization, and manual quantization.

---

## 📌 Objective

To demonstrate a reproducible and automated MLOps pipeline including:

- Model training using `LinearRegression` (scikit-learn)
- Model testing and validation using `pytest`
- Manual quantization of model parameters
- Docker container to deploy inference
- CI/CD using GitHub Actions for automated testing, training, and deployment

---

## 🗂️ Project Structure

```
├── src/
│   ├── train.py          # Train and save model
│   ├── quantize.py       # Manual quantization of model parameters
│   ├── predict.py        # Load model and make predictions
│   └── utils.py          # (Optional) helper functions
├── tests/
│   └── test_train.py     # Unit tests for training pipeline
├── .github/workflows/
│   └── ci.yml            # CI/CD pipeline (GitHub Actions)
├── model.joblib          # Trained model artifact
├── Dockerfile            # Docker setup for running predict
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions (Local)

```bash
# Clone the Repository
git clone https://github.com/<your-username>/mlops-assignment.git
cd mlops-assignment

# Setup Environment
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate

# Install Requirements
pip install -r requirements.txt

# Run Pipeline Manually
python src/train.py
python src/quantize.py
python src/predict.py

# Run Unit Tests
pytest tests/

# Docker Build & Run
docker build -t mlops-linear .
docker run mlops-linear
```

---

## 🔁 CI/CD with GitHub Actions

This repository uses GitHub Actions to automatically:

| Job Name                | Description                                                           | Trigger                |
|-------------------------|-----------------------------------------------------------------------|------------------------|
| 🧪 `test-suite`          | Runs `pytest` on push to `main`                                       | Push to `main`         |
| 🛠️ `train-and-quantize` | Trains model and quantizes parameters                                 | After test-suite       |
| 🐳 `build-and-test-container` | Builds Docker image and verifies prediction using `predict.py`   | After quantization     |

> CI/CD config is located in: `.github/workflows/ci.yml`

---

## 📊 Comparison Table

| Metric               | Unquantized Parameters     | Quantized Parameters (`uint8`) |
|----------------------|----------------------------|-------------------------------|
| Parameter File Size  | ~8 KB                      | ~1 KB                         |
| Data Type            | `float64`                  | `uint8`                       |
| Range                | Real-valued (-∞ to ∞)      | [0, 255]                      |
| Precision            | High                       | Lower (manual approximation) |
| Use Case             | Accurate model inference   | Lightweight inference         |

---

## 📬 Author & Submission

- **Submitted by**: Gitansh Syal  
- **Course**: MLOps  
- **Instructor**: Dr. Pratik Mazumder  
- **GitHub Repo**: https://github.com/Gitanshsyal3952/MajorAssignment.git
