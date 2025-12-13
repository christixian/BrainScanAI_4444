from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import os
import uuid
from datetime import datetime
from database import init_db, add_prediction, get_history, clear_history
from gradcam import GradCAM, generate_heatmap_overlay, get_base64_overlay

# Define Model Architecture (Must match training)
class BrainCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files for Uploads
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize DB
init_db()

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainCNN(num_classes=4).to(device)
MODEL_PATH = "models/brain_cnn_4class.pth" # Moved to models folder

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using initialized model (random weights) - Training might be in progress.")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")

model.eval()

# GradCAM
# Target the last convolutional layer. 
# In features: 
# 0: Conv, 1: BN, 2: ReLU, 3: Pool
# 4: Conv, 5: BN, 6: ReLU, 7: Pool
# 8: Conv, 9: BN, 10: ReLU, 11: Pool
# 12: Conv, 13: BN, 14: ReLU, 15: Pool
# Target the Conv layer (index 12) to avoid in-place ReLU errors with Grad-CAM
target_layer = model.features[12]
grad_cam = GradCAM(model, target_layer)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)), # Match training size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]), # Match training normalization
])

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
UNCERTAIN_THRESHOLD_HEALTHY = 0.55
UNCERTAIN_THRESHOLD_UNHEALTHY = 0.5

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validate File Size (Max 20MB)
    MAX_FILE_SIZE = 20 * 1024 * 1024 # 20MB
    
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read file")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (Max 20MB)")

    # 2. Validate Image Format & Convert
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    # Save Image
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join("static/uploads", filename)
    img.save(filepath)
    image_url = f"http://localhost:8000/static/uploads/{filename}"

    # 3. Preprocess & Inference
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        top_confidence = float(probabilities[predicted_index])

    # 4. Binary mapping based on aggregated tumor vs healthy probability
    healthy_probability = float(probabilities[class_names.index("notumor")])
    tumor_probability = float(sum(probabilities[i] for i, name in enumerate(class_names) if name != "notumor"))

    if tumor_probability >= healthy_probability:
        binary_label = "unhealthy"
        binary_conf = tumor_probability
    else:
        binary_label = "healthy"
        binary_conf = healthy_probability

    # Flag low-confidence predictions explicitly
    unc_threshold_use = UNCERTAIN_THRESHOLD_HEALTHY if binary_label=="healthy" else UNCERTAIN_THRESHOLD_UNHEALTHY
    is_uncertain = top_confidence < unc_threshold_use
    prediction_4class = "uncertain" if is_uncertain else predicted_class

    # 5. Generate Heatmap (Grad-CAM)
    # Need gradients, so ensure requires_grad is handled by GradCAM or we might need to enable it if it was disabled globally?
    # torch.no_grad() context above prevents gradients. We need to run GradCAM outside it or ensure it handles it.
    # The GradCAM class usually does a forward pass with gradients.
    
    heatmap = grad_cam(tensor, class_idx=predicted_index)
    
    # Create overlay
    img_resized = img.resize((128, 128)) # Resize to match model input for overlay
    overlay_img = generate_heatmap_overlay(img_resized, heatmap)
    heatmap_base64 = "data:image/png;base64," + get_base64_overlay(img_resized, overlay_img)

    # 6. Save to History
    confidence_scores = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }

    add_prediction(
        prediction_4class=prediction_4class,
        prediction_binary=binary_label,
        confidence_scores=confidence_scores,
        binary_confidence=float(binary_conf),
        heatmap_base64=heatmap_base64,
        image_url=image_url
    )


    return {
        "prediction_4class": prediction_4class,
        "prediction_binary": binary_label,
        "confidence_scores": confidence_scores,
        "binary_confidence": float(binary_conf),
        "top_class": predicted_class,
        "top_class_confidence": top_confidence,
        "uncertain_threshold": unc_threshold_use,
        "heatmap_base64": heatmap_base64,
        "image_url": image_url
    }

@app.get("/history")
async def get_history_endpoint():
    return get_history()

@app.delete("/history")
async def clear_history_endpoint():
    clear_history()
    return {"message": "History cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)