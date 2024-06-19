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
    filename = os.path.basename(img_path).split(".")[0]
    return {"filename": filename, "probabilities": result}


def format_output(output):
    return f"{output['filename']}", output["probabilities"]


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
    fn=lambda img: format_output(classify_image(img)),
    inputs=gr.Image(type="filepath"),
    outputs=[gr.Textbox(label="True Label (from filename)"), gr.Label()],
    examples=examples,
    title="Pneumonia X-Ray 3-Class Classification with Vision Transformer (ViT) using data augmentation",
    description="Upload an X-ray image to classify it as normal, viral or bacterial pneumonia. Checkout the model in more details [here](https://huggingface.co/pawlo2013/vit-pneumonia-x-ray_3_class). The examples presented are take from the test set of [Kermany et al. (2018) dataset.](https://data.mendeley.com/datasets/rscbjbr9sj/2)",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
