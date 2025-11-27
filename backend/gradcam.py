import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Generate Heatmap
        gradients = self.gradients[0]
        pooled_gradients = torch.mean(gradients, dim=[1, 2])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).cpu().detach()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        
        return heatmap.numpy()

def generate_heatmap_overlay(original_img_pil, heatmap_np):
    # Resize heatmap to match original image
    heatmap_np = Image.fromarray(np.uint8(255 * heatmap_np))
    heatmap_np = heatmap_np.resize(original_img_pil.size, resample=Image.BICUBIC)
    
    # Apply colormap (Jet)
    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(np.array(heatmap_np) / 255.0) # RGBA
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8) # RGB
    
    return Image.fromarray(heatmap_colored)

def get_base64_overlay(original_img, heatmap_overlay, alpha=0.4):
    # Blend images
    original_img = original_img.convert("RGB")
    blended = Image.blend(original_img, heatmap_overlay, alpha=alpha)
    
    # Convert to base64
    buffered = io.BytesIO()
    blended.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
