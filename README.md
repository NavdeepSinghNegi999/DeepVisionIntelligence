# DeepVisionIntelligence 🧠📸

## 🌟 Overview

**DeepVisionIntelligence** is a **Transformer-based image captioning system** that generates natural language descriptions from images.

The model combines **CNN-based visual feature extraction (EfficientNetB0)** with a **Transformer decoder** to learn rich multimodal representations. It performs **sequence modeling** to generate captions word-by-word, conditioned on image features, producing contextually accurate descriptions such as “a dog playing in the park” or “a plate of pasta on a wooden table”.

Trained on **8,000+ images**, the model leverages **transfer learning, attention mechanisms, and token-level prediction** to understand visual context and convert it into fluent text.

Beyond model development, the system is deployed as a **real-time application** using FastAPI and Google Cloud. An end-to-end **MLOps pipeline** is integrated for experiment tracking, data versioning, and automated deployment via CI/CD, ensuring the model is scalable, reproducible, and production-ready.

---
### ✨ Key Features

- 🤖 **AI-Powered Caption Generation** - Transformer-based model generates context-aware descriptions  
- 🧠 **Deep Learning Architecture** - CNN (EfficientNet) + Transformer for vision-language learning  
- ⚡ **Real-time Inference** - Fast caption generation using TensorFlow  
- 🖼️ **Image Upload Interface** - Simple UI for testing model predictions  
- ⚙️ **Production-ready Deployment** - FastAPI + Docker + Cloud Run  
- 🔁 **CI/CD Pipeline** - Automated build and deployment using GitHub Actions  
- 📊 **Experiment Tracking & Versioning** - DVC and MLflow for reproducibility  
- ☁️ **Cloud Integration** - Google Cloud for scalable storage and deployment  

## 🧠 Deep Learning Highlights

- Transformer-based sequence generation for image captioning  
- Multi-modal learning (vision + language fusion)  
- Pretrained CNN (EfficientNet) for feature extraction  
- Attention mechanism for context-aware caption generation  
- Tokenization and sequence modeling for NLP  
- Trained using Sparse Categorical Crossentropy loss and optimized with Adam  
- Evaluated using BLEU score metrics  

### 🔍 Model Details

- **Encoder**:
  - Pretrained EfficientNetB0 extracts high-level visual features from images
  - Transfer learning is used to leverage ImageNet representations
  - Extracted features are projected into the embedding space for the Transformer

- **Decoder**:
  - Transformer-based decoder with multi-head self-attention
  - Learns contextual relationships between words using sequence modeling
  - Generates captions token-by-token conditioned on image features

- **Training**:
  - Loss Function: Sparse Categorical Crossentropy (token-level prediction)
  - Padding tokens are masked during loss computation (`reduction="none"`)
  - Optimizer: Adam with learning rate warmup schedule
  - Captions are tokenized, indexed, and padded to fixed sequence length

- **Inference**:
  - Given an image, the encoder extracts feature embeddings
  - The decoder generates captions sequentially using previous tokens as context
  - At each step, the next word is selected based on predicted probability distribution

---

## 🚀 Deployment & Production

- Deployed using FastAPI for real-time inference  
- Containerized using Docker for reproducibility  
- Integrated with Google Cloud Run for scalable deployment  
- CI/CD pipeline using GitHub Actions for automated builds and deployment  
- DVC and MLflow used for experiment tracking and model versioning  

This ensures the model is not just trained but also production-ready and scalable.

---

## 🚀 Demo

| |
|:---:|
| **Front Webpage** |
| ![Upload](tests/screenshots/5_deploy_webapp.png) |
| **AI Caption Result** |
| ![Result](tests/screenshots/6_deploy_webpage_result.png) |

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Deep Learning** | TensorFlow, Transformer (Encoder-Decoder), EfficientNetB0 | Image captioning using vision-language modeling and attention |
| **Sequence Modeling / NLP** | Tokenization, Text Processing, NLTK | Caption generation and sequence handling |
| **Model Training** | Adam Optimizer, Learning Rate Warmup, Sparse Categorical Crossentropy | Training and optimization of sequence prediction model |
| **MLOps** | DVC, MLflow, Docker, Git | Experiment tracking, data versioning, reproducibility |
| **CI/CD** | GitHub Actions | Automated build and deployment pipeline |
| **Backend** | FastAPI | Real-time model inference API |
| **Frontend** | HTML, CSS, JavaScript | User interface for image upload and caption display |
| **Cloud Platform** | Google Cloud Platform (GCP) | Cloud Run deployment and Cloud Storage for artifacts |


---

## 📋 Prerequisites

- Python 3.10
- pip (Python package manager)
- Git (optional, for cloning)
- Google Cloud account (optional, for remote storage)

---

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/NavdeepSinghNegi999/DeepVisionIntelligence.git
cd DeepVisionIntelligence
```

### 2. Create Virtual Environment
```bash
# Create conda environment with Python 3.10
conda create -n deepvision python=3.10.0

# Install CUDA and cuDNN for GPU support
conda install -n deepvision cudatoolkit=11.2 cudnn=8.1 -c conda-forge

# Activate the environment
conda activate deepvision
```

### 3. Install Dependencies
```bash
pip install -r requirements/all_requirements.txt
```

### 4. Set Up DVC (Optional for Cloud Storage)
```bash
# Initialize DVC
dvc init

# Add remote storage (example: local)
dvc remote add -d localremote ./dvc-storage

# Or for Google Cloud
# dvc remote add -d gcsremote gs://your-bucket-name
```

---

## 🚀 Usage

### Start the Application
```bash
uvicorn main:app --reload
```

### Access the Web Interface
Open your browser and navigate to:
```
http://localhost:8080
```


---

## 📁 Project Structure

```
deepvisionintelligence/
├── .dvc/                       # DVC configuration
├── data/                       # Dataset files
├── logs/                       # Training logs
├── mlruns/                     # MLflow experiment tracking
├── notebooks/                  # Jupyter notebooks
├── requirements/               # Dependency files
├── src/                        # Source code
│   ├── data_component/        # Data processing modules
│   ├── evaluation/            # Evaluation metrics
│   ├── inference/             # Inference modules
│   ├── models/                # Model architectures
│   ├── training/              # Training modules
│   └── utils/                  # Utility functions
├── tests/                      # Test screenshots
│   └── screenshots/            # Predicted images and test results
├── .dockerignore
├── .gitignore
├── artifacts.dvc               # DVC tracked artifacts
├── config.yaml                 # Configuration file
├── data.dvc                    # DVC tracked data
├── Dockerfile                   # Docker configuration
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference script
├── logs.dvc                     # DVC tracked logs
├── main.py                      # FastAPI application
├── README.md                    
└── train.py                     # Training script
```

---

## 🧠 Model Architecture

DeepVisionIntelligence uses a Transformer encoder-decoder architecture:

```

        📸                   🧠                    📊      
   Image Input    ───▶    Encoder        ───▶   Feature Vector  
                               

                                                     │
                                                     ▼

        📝                    🤖                   🔤      
   Caption Output    ◀───   Decoder   ◀───   Sequence Embedding   
                  

```

- **Encoder**: EfficientNetB0 (pre-trained on ImageNet)
- **Decoder**:  Transformer with multi-head self-attention

---

## 📊 DVC Integration

DeepVisionIntelligence uses DVC for data versioning:

```bash
# Track folders
dvc add data/
dvc add mlruns/
dvc add artifacts/

# Push to remote storage
dvc push

# Pull from remote storage
dvc pull

# Check status
dvc status
```

---

## 🎯 Performance Metrics

| Metric | Score  |
|--------|--------|
| BLEU-1 | 0.9432 |
| BLEU-2 | 0.9029 |
| BLEU-3 | 0.8640 |
| BLEU-4 | 0.8085 |



---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🎯 Open a Pull Request

---

## 👏 Acknowledgments

- TensorFlow team for the amazing framework
- FastAPI for the elegant web framework
- DVC community for data versioning tools

---

## 📧 Contact

**Name** - Navdeep Singh

**LinkedIn ID** - https://www.linkedin.com/in/navdeep-singh-n/

---

<div align="center">
  
**Built by Navdeep Singh**

[⬆ Back to Top](#deepvisionintelligence-)



</div>



