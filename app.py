import os
import gradio as gr
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
def classify_and_visualize(img, device="cpu", discard_ratio=0.9, head_fusion="mean"):
    img = img.convert("RGB")
    processed_input = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**processed_input)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].tolist()
        prediction = torch.argmax(logits, dim=-1).item()
        predicted_class = class_names[prediction]

    result = {class_name: prob for class_name, prob in zip(class_names, probabilities)}

    # Generate attention heatmap
    heatmap_img = show_final_layer_attention_maps(
        model, processed_input, device, discard_ratio, head_fusion
    )

    return {"probabilities": result, "heatmap": heatmap_img}


def format_output(output):
    return (
        output["probabilities"],
        output["heatmap"] if output["heatmap"] is not None else None,
    )


# Function to load examples from a folder
def load_examples_from_folder(folder_path):
    examples = []
    for file in os.listdir(folder_path):
        if file.endswith((".png", ".jpg", ".jpeg")):
            examples.append(Image.open(os.path.join(folder_path, file)))
    return examples


# Function to show final layer attention maps
def show_final_layer_attention_maps(
    model, tensor, device, discard_ratio=0.6, head_fusion="max", only_last_layer=False
):
    image = tensor["pixel_values"].to(device).squeeze(0)

    with torch.no_grad():
        outputs = model(**tensor, output_attentions=True)

        if outputs.attentions[0] is None:
            print("Attention outputs are None.")
            return None

        image = image - image.min()
        image = image / image.max()

        result = torch.eye(outputs.attentions[0].size(-1)).to(device)
        if only_last_layer:
            attention_list = outputs.attentions[-1].unsqueeze(0).to(device)
        else:
            attention_list = outputs.attentions

        for attention in attention_list:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).cpu().numpy()
        mask = mask / np.max(mask)

        mask = cv2.resize(mask, (224, 224))

        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        heatmap = plt.cm.jet(mask)[:, :, :3]

        showed_img = image.permute(1, 2, 0).detach().cpu().numpy()
        showed_img = (showed_img - np.min(showed_img)) / (
            np.max(showed_img) - np.min(showed_img)
        )
        superimposed_img = heatmap * 0.4 + showed_img * 0.6

        superimposed_img_pil = Image.fromarray(
            (superimposed_img * 255).astype(np.uint8)
        )

        return superimposed_img_pil


# Define the path to the examples folder
examples_folder = "./examples"
examples = load_examples_from_folder(examples_folder)

# Create the Gradio interface
iface = gr.Interface(
    fn=lambda img: format_output(classify_and_visualize(img)),
    inputs=gr.Image(type="pil", label="Upload X-Ray Image"),
    outputs=[
        gr.Label(),
        gr.Image(label="Attention Heatmap"),
    ],
    examples=examples,
    title="Pneumonia X-Ray 3-Class Classification with Vision Transformer (ViT) using data augmentation",
    description="Upload an X-ray image to classify it as normal, viral or bacterial pneumonia. Checkout the model in more details [here](https://huggingface.co/pawlo2013/vit-pneumonia-x-ray_3_class). The examples presented are taken from the test set of [Kermany et al. (2018) dataset.](https://data.mendeley.com/datasets/rscbjbr9sj/2.) The attention heatmap over all layers of the transfomer done by the attention rollout techinique by the implementation of [jacobgil](https://github.com/jacobgil/vit-explain).",
)
# Launch the app
if __name__ == "__main__":
    iface.launch(debug=True)
