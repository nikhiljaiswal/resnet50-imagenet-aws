import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from train_resnet50 import ResNet, Bottleneck

# Load ImageNet class labels
try:
    with open("imagenet_classes.json", "r") as f:
        class_labels = json.load(f)
    print(f"Loaded {len(class_labels)} class labels")
except FileNotFoundError:
    print("Warning: imagenet_classes.json not found, creating simplified labels")
    # Fallback to a simplified version
    class_labels = {str(i): f"class_{i}" for i in range(1000)}
except json.JSONDecodeError:
    print("Warning: Error parsing imagenet_classes.json, using simplified labels")
    class_labels = {str(i): f"class_{i}" for i in range(1000)}
except Exception as e:
    print(f"Warning: Unexpected error loading class labels: {e}")
    class_labels = {str(i): f"class_{i}" for i in range(1000)}


def create_model():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


def load_model(model_path):
    model = create_model()
    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # Handle DataParallel/DDP state dict
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Loading pretrained ResNet50 as fallback...")
        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
        model.eval()
        return model


# Preprocessing transform
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Global variable for model
global_model = None


def predict(image):
    global global_model

    # Load model only once
    if global_model is None:
        try:
            global_model = load_model(
                "/data/nikhil_workspace/imagenet_workspace/runs/resnet50_20241227_010125/best_model.pth"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    # Preprocess image
    if image is None:
        return None

    try:
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = global_model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 5 predictions
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # Create results dictionary
        results = []
        for i in range(5):
            class_idx = top5_catid[i].item()
            # Use list indexing instead of dictionary get()
            class_label = (
                class_labels[class_idx]
                if class_idx < len(class_labels)
                else f"class_{class_idx}"
            )
            results.append(
                {
                    "label": class_label,
                    "class_id": class_idx,
                    "confidence": float(top5_prob[i].item()),
                }
            )

        return results
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(f"Class indices: {[idx.item() for idx in top5_catid]}")  # Debug info
        return None


# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.JSON(),
    title="ResNet50 ImageNet Classifier",
    description="Upload an image to get top-5 predictions from our trained ResNet50 model.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)
