"""
Visualization script for 3-class pneumonia classification.
Creates a single image showing all 3 classes with predictions and attention heatmaps.
"""

import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset

# Model and processor configuration
model_name_or_path = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

# Load dataset to get correct class names
dataset_path = "pawlo2013/chest_xray"
train_dataset = load_dataset(dataset_path, split="train")
class_names = train_dataset.features["label"].names
print(f"Class names from dataset: {class_names}")

# Load ViT model
model = ViTForImageClassification.from_pretrained(
    "./models",
    num_labels=len(class_names),
    id2label={str(i): label for i, label in enumerate(class_names)},
    label2id={label: i for i, label in enumerate(class_names)},
)

# Set model to evaluation mode
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def show_final_layer_attention_maps(
    outputs,
    processed_input,
    device,
    discard_ratio=0.6,
    head_fusion="max",
    only_last_layer=False,
):
    """Generate attention heatmap from model outputs."""
    
    with torch.no_grad():
        # Extract pixel_values from the processed input
        if hasattr(processed_input, 'pixel_values'):
            image = processed_input.pixel_values.squeeze(0).to(device)
        else:
            image = processed_input.squeeze(0)
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


def classify_and_visualize(img_path):
    """Classify an image and return predictions with heatmap."""
    img = Image.open(img_path).convert("RGB")
    processed_input = processor(images=img, return_tensors="pt")
    pixel_values = processed_input.pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].tolist()
        prediction = torch.argmax(logits, dim=-1).item()

    result = {class_name: prob for class_name, prob in zip(class_names, probabilities)}
    
    # Generate attention heatmap
    heatmap_img = show_final_layer_attention_maps(
        outputs, processed_input, device, discard_ratio=0.9, head_fusion="mean"
    )

    return {"probabilities": result, "heatmap": heatmap_img, "original": img}


def create_3class_visualization(example_folder="./examples", output_path="./3class_visualization.png"):
    """
    Create a single visualization showing all 3 classes.
    """
    global class_names
    
    # Process each class
    results = {}
    for class_name in class_names:
        img_path = os.path.join(example_folder, f"{class_name}.jpeg")
        if os.path.exists(img_path):
            results[class_name] = classify_and_visualize(img_path)
            print(f"Processed {class_name}: {results[class_name]['probabilities']}")
        else:
            print(f"Warning: {img_path} not found")
    
    if len(results) == 0:
        print("No images found to process!")
        return
    
    # Create figure with 3 rows (one per class) and 3 columns (original, heatmap, bar chart)
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Pneumonia X-Ray Classification - 3 Class Visualization", fontsize=16, fontweight='bold')
    
    # Color scheme for classes (based on class name)
    def get_class_color(class_name):
        if "Normal" in class_name:
            return "#2ecc71"      # Green
        elif "Bacterial" in class_name:
            return "#e74c3c"      # Red
        elif "Viral" in class_name:
            return "#3498db"      # Blue
        else:
            return "#95a5a6"      # Gray
    
    # Map file names to class indices
    class_to_file = {}
    for idx, class_name in enumerate(class_names):
        class_to_file[class_name] = os.path.join(example_folder, f"{class_name}.jpeg")
    
    for idx, class_name in enumerate(class_names):
        if class_name not in results:
            continue
            
        data = results[class_name]
        probs = data["probabilities"]
        original_img = np.array(data["original"])
        heatmap_img = np.array(data["heatmap"])
        color = get_class_color(class_name)
        
        # Column 1: Original Image
        ax1 = fig.add_subplot(3, 3, idx * 3 + 1)
        ax1.imshow(original_img)
        ax1.set_title(f"{class_name} - Original X-Ray", fontsize=12, fontweight='bold', color=color)
        ax1.axis('off')
        
        # Column 2: Attention Heatmap
        ax2 = fig.add_subplot(3, 3, idx * 3 + 2)
        ax2.imshow(heatmap_img)
        ax2.set_title(f"Attention Heatmap", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Column 3: Prediction Probabilities (Bar Chart)
        ax3 = fig.add_subplot(3, 3, idx * 3 + 3)
        labels = list(probs.keys())
        values = list(probs.values())
        colors = [get_class_color(label) for label in labels]
        
        bars = ax3.barh(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probability", fontsize=11)
        ax3.set_title(f"Class Probabilities", fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.2%}', va='center', fontsize=10, fontweight='bold')
        
        # Highlight predicted class
        predicted_class = max(probs, key=probs.get)
        ax3.set_facecolor('#f8f9fa')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()
    
    return output_path


if __name__ == "__main__":
    create_3class_visualization()
