import os
import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset

# Model and processor configuration
model_name_or_path = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

# Load dataset (adjust dataset_path accordingly)
dataset_path = "pawlo2013/chest_xray"
train_dataset = load_dataset(dataset_path, split="train")
class_names = train_dataset.features["label"].names

# Load ViT model
model = ViTForImageClassification.from_pretrained(
    "./models",
    num_labels=len(class_names),
    id2label={str(i): label for i, label in enumerate(class_names)},
    label2id={label: i for i, label in enumerate(class_names)},
)

# Set model to evaluation mode
model.eval()

# Define transformation for incoming images


# Function to predict on a single image
def classify_image(img):
    img = processor(img.convert("RGB"))  # Apply ViT processor
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)  # Forward pass through the model
        _, predicted = torch.max(output, 1)  # Get predicted class index
    return class_names[predicted.item()]  # Return predicted class label


# Function to process all images in a folder
def classify_all_images():
    examples_dir = "examples"
    results = []
    for filename in os.listdir(examples_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(examples_dir, filename)
            img = Image.open(img_path)
            img = processor(img.convert("RGB"))  # Apply ViT processor
            img = img.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
            results.append(
                (filename, class_names[predicted.item()])
            )  # Store filename and predicted class label
    return results


# Create Gradio interface for single image classification
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.inputs.Image(type="pil", label="Upload Image"),
    outputs=gr.outputs.Label(num_top_classes=3),
    title="Image Classification",
    description="Classifies an image into one of the predefined classes.",
)

# Create Gradio interface for all images classification
iface_all_images = gr.Interface(
    fn=classify_all_images,
    inputs=None,
    outputs=gr.outputs.Label(type="key_values", label="Image Classifications"),
    title="Batch Image Classification",
    description="Classifies all images in the 'examples' folder.",
)

# Launch both interfaces
iface.launch(share=True)
iface_all_images.launch(share=True)
