import os
import gradio as gr
from PIL import Image
import torch
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


# Define the classification function
def classify_image(img_path):
    img = Image.open(img_path)
    processed_input = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**processed_input)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].tolist()

    result = {class_name: prob for class_name, prob in zip(class_names, probabilities)}
    filename = os.path.basename(img_path)
    return {"filename": filename, "probabilities": result}


def format_output(output):
    return f"Filename: {output['filename']}", output["probabilities"]


# Function to load examples from a folder
def load_examples_from_folder(folder_path):
    examples = []
    for file in os.listdir(folder_path):
        if file.endswith((".png", ".jpg", ".jpeg")):
            examples.append(os.path.join(folder_path, file))
    return examples


# Define the path to the examples folder
examples_folder = "./examples"
examples = load_examples_from_folder(examples_folder)

# Create the Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Textbox(label="Filename"), gr.BarChart(label="Class Probabilities")],
    examples=examples,
    title="Pneumonia X-Ray Classification",
    description="Upload an X-ray image to classify it as normal or pneumonia.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
