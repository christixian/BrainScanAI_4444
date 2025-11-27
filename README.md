
# BrainScan AI

> **Advanced Deep Learning for MRI Tumor Detection**

BrainScan AI is a research prototype designed to assist in the classification of brain tumors from MRI scans. It utilizes a custom Convolutional Neural Network (CNN) to detect and classify tumors into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

The application features a modern, responsive web interface built with Next.js and a robust FastAPI backend that serves the PyTorch model. It also includes **Grad-CAM** (Gradient-weighted Class Activation Mapping) visualization to highlight the specific regions of the MRI that influenced the model's decision, providing explainable AI capabilities.

---

## Key Features

*   **Real-time Analysis**: Instant classification of uploaded MRI scans.
*   **4-Class Detection**: Distinguishes between Glioma, Meningioma, Pituitary tumors, and Healthy brains.
*   **Explainable AI (XAI)**: Generates **Grad-CAM heatmaps** to visualize tumor location.
*   **History Tracking**: Automatically saves past predictions to a local database for review.
*   **Privacy-Focused**: All processing happens locally on your machine.

---

## Tech Stack

### **Frontend**
*   **Framework**: Next.js 16 (React 19)
*   **Styling**: Tailwind CSS
*   **Icons**: Lucide React
*   **Language**: TypeScript

### **Backend**
*   **Framework**: FastAPI
*   **ML Engine**: PyTorch
*   **Image Processing**: Pillow (PIL), NumPy
*   **Database**: SQLite (for tracking uploaded history and predictions)

---

## Project Structure


```bash
AI_PROJECT2/
├── backend/                 # Python API & Model Logic
│   ├── main.py              # FastAPI entry point & endpoints
│   ├── database.py          # SQLite database operations
│   ├── gradcam.py           # Grad-CAM visualization logic
│   ├── requirements.txt     # Python dependencies
│   ├── models/              # Trained models
│   │   └── brain_cnn_4class.pth
│   ├── training/            # Training scripts
│   │   └── train_brain_cnn_pytorch.py
│   ├── utils/               # Utility scripts
│   │   └── clean_duplicates.py
│   ├── static/uploads/      # Temp storage for uploaded images
│   └── brainscan_history.db # Local database file (auto-created)
│
├── frontend/                # Next.js Web Application
│   ├── src/app/             # Pages & Layouts
│   ├── src/components/      # UI Components (UploadArea, ResultDisplay)
│   └── public/              # Static assets
│
└── README.md                # This file

```

---

## Model Architecture

The core of BrainScan AI is a custom **Convolutional Neural Network (CNN)** designed to identify patterns in MRI scans. It processes images in two main stages:

1.  **Feature Extraction (The "Eyes")**:
    *   The network uses **4 Convolutional Blocks** to break down the image.
    *   **Convolutional Layers (`Conv2d`)**: Scan the image to detect features like edges, curves, and textures. The network starts with 32 filters and increases to 256, allowing it to learn increasingly complex patterns (from simple lines to tumor shapes).
    *   **Batch Normalization (`BatchNorm`)**: Stabilizes learning by normalizing inputs between layers.
    *   **ReLU Activation**: Adds non-linearity, allowing the model to learn complex relationships (essentially deciding if a feature is "present" or not).
    *   **Max Pooling (`MaxPool2d`)**: Reduces the image size by half after each block (128x128 → 64x64 → ... → 8x8), keeping only the most important features and reducing computation.

2.  **Classification (The "Brain")**:
    *   **Flatten**: Converts the 2D feature maps (256 channels x 8x8) into a single 1D vector of 16,384 values.
    *   **Fully Connected Layer (`Linear`)**: Analyzes the combined features to form a high-level understanding.
    *   **Dropout (0.5)**: Randomly turns off 50% of neurons during training to prevent the model from memorizing the data (overfitting), forcing it to generalize better.
    *   **Output Layer**: Produces 4 final scores (logits), one for each class: Glioma, Meningioma, Pituitary, and No Tumor.

---

## Data Preprocessing

The original dataset contained several duplicate images, which could bias the training process. The dataset used for this project is the **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)** from Kaggle.

*   **`backend/utils/clean_duplicates.py`**: A utility script was created to sanitize the dataset.
    *   It calculates **MD5 hashes** for every image in the `Training` and `Testing` directories.
    *   It automatically detects and **removes duplicate files** to ensure data integrity.

---

## Training from Scratch

If you want to retrain the model yourself using the Kaggle dataset:

1.  **Download Data**:
    *   Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data).
    *   Extract the `Training` and `Testing` folders into the root of this project (`AI_PROJECT2/`).
2.  **Run Training Script**:
    ```bash
    python backend/training/train_brain_cnn_pytorch.py
    ```
    *   The script will train for 20 epochs and save the best model as `brain_cnn_4class.pth` in the root directory.
3.  **Deploy Model**:
    *   Move the generated `brain_cnn_4class.pth` file into the `backend/models/` directory.
    *   Restart the backend server to load the new model.

---

## Getting Started

Follow these steps to run the project locally on your machine.

### 1. Prerequisites
*   **Python 3.8+** installed.
*   **Node.js 18+** installed.

### 2. Backend Setup
The backend runs the AI model.

1.  Open a terminal in the root folder (`AI_PROJECT2`).
2.  Navigate to the backend folder:
    ```bash
    cd backend
    ```
3.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Start the server:
    ```bash
    python main.py
    ```
    *The backend will start at `http://localhost:8000`.*

### 3. Frontend Setup
The frontend is the website you interact with.

1.  Open a **new** terminal window (keep the backend running).
2.  Navigate to the frontend folder:
    ```bash
    cd frontend
    ```
3.  Install Node dependencies:
    ```bash
    npm install
    ```
4.  Start the development server:
    ```bash
    npm run dev
    ```
5.  Open your browser and go to: **`http://localhost:3000`**

---

## How to Test

1.  **Launch the App**: Ensure both Backend (port 8000) and Frontend (port 3000) are running.
2.  **Upload an MRI from Testing Folder**:
    *   Go to `http://localhost:3000`.
    *   Click the upload area or drag & drop a brain MRI image (JPG/PNG).
    *   *Note: You can find test images online by searching for "[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data)".*
3.  **View Results**:
    *   The AI will analyze the image in seconds.
    *   You will see the **Predicted Class** (e.g., Meningioma) and a **Confidence Score**.
    *   A **Heatmap Overlay** will appear, showing where the tumor is located.
4.  **Check History**:
    *   Click the "History" tab in the navigation bar to see past predictions.

---

## Disclaimer

**Research Prototype Only**: This tool is for educational and research purposes. It is **not** a medical device and should **not** be used for actual medical diagnosis.
