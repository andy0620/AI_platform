import argparse
import torch
import torch.nn as nn
from transformers import AutoModel

# This mapping helps translate the user-friendly names from the UI
# into actual HuggingFace model identifiers.
# This mapping translates user-friendly names from the UI into the exact
# HuggingFace model identifiers found in the documentation.
MODEL_NAME_MAP = {
    # NOTE FOR USER: The DINOv3 models are gated. To run them, you must be
    # logged into a HuggingFace account with access. For demonstration purposes,
    # ViT-B/16 is temporarily mapped to a public Google ViT model.
    "ViT-S/16 (21M)": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "ViT-B/16": "google/vit-base-patch16-224-in21k", # Using a public model for demonstration
    "ViT-L/16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "ViT-7B/16 (7B)": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "ConvNeXt-T": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "ConvNeXt-S": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "ConvNeXt-B": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "ConvNeXt-L": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}

class DINOv3Classifier(nn.Module):
    """
    DINOv3 Classifier model based on the definition in dev_doc.md.
    The backbone is frozen, and a new classification head is trained.
    """
    def __init__(self, model_name="facebook/dinov3-vitb16", num_classes=10):
        super().__init__()
        print(f"Loading backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        # Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        print("DINOv3Classifier initialized successfully.")

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            # Using pooler_output as per the dev_doc
            pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

def main():
    """
    Main function to set up and run the training process.
    """
    parser = argparse.ArgumentParser(description="Train a DINOv3 Classifier.")
    parser.add_argument('--model-size', type=str, required=True, choices=MODEL_NAME_MAP.keys(),
                        help='The size of the DINOv3 model to use.')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes for the classifier.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs.')

    args = parser.parse_args()
    print(f"--- Starting Training Script ---")
    print(f"Model Size: {args.model_size}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}")

    # Get the full model name from the mapping
    model_name = MODEL_NAME_MAP.get(args.model_size)
    if not model_name:
        print(f"Error: Invalid model size '{args.model_size}'")
        return

    try:
        model = DINOv3Classifier(model_name=model_name, num_classes=args.num_classes)
        print("\nScript setup complete. Starting training loop...")
    except Exception as e:
        print(f"\nAn error occurred during model initialization: {e}")
        print("This might be due to an invalid model name or network issues.")
        print("Please check the HuggingFace model name and your internet connection.")
        return

    # --- Simplified Training Loop with Dummy Data ---
    # 1. Create a dummy dataset
    print("\n1. Creating dummy dataset...")
    # Batch size of 4, 3 channels, 224x224 images
    dummy_images = torch.randn(16, 3, 224, 224)
    dummy_labels = torch.randint(0, args.num_classes, (16,))
    dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
    print("Dummy dataset and dataloader created.")

    # 2. Setup optimizer and loss function
    print("\n2. Setting up optimizer and loss function...")
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    print("Optimizer and loss function are set up.")

    # 3. The Training Loop
    print("\n3. Starting training loop...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print("\n--- Training Loop Finished ---")
    print(f"--- Script Finished ---")

if __name__ == "__main__":
    main()
