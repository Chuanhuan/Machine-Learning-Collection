# --- START OF FILE vit_implementation.py ---

import torch

# import pandas as pd # No longer needed
from torch import nn
from torch import optim
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # Dataset not strictly needed now, but kept for potential future custom use

# from torchvision import transforms # Redundant, imported below

import torchvision.transforms as transforms
from torchvision.datasets import MNIST  # Import MNIST dataset

# from sklearn.model_selection import train_test_split # No longer needed
import matplotlib.pyplot as plt
import numpy as np
import random
import timeit
from tqdm import tqdm

# |%%--%%| <G7PznAvySN|klOAPYSc5g> # (Keep cell markers if using an interactive environment like VS Code Notebooks)

RANDOM_SEED = 42
BATCH_SIZE = 512
EPOCHS = 15  # Reduced epochs for quicker demonstration, adjust as needed
LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 4  # 32 // 4 = 8 patches per side
IMG_SIZE = 32
IN_CHANNELS = 1
NUM_HEADS = 8
DROPOUT = 0.1  # Slightly increased dropout as per some common ViT practices
HIDDEN_DIM = 768  # Note: Standard ViT often uses hidden_dim = embed_dim * 4, but keeping original for now
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE**2) * IN_CHANNELS  # 4*4*1 = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # (28//4)**2 = 7**2 = 49

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Define data directory
DATA_ROOT = "~/Documents/data"

# |%%--%%| <klOAPYSc5g|sEei6hocnX>


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Model Definition (as provided) ---
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        # Ensure the hub model is available or handle potential errors
        try:
            # Load the specific ResNet56 model trained on CIFAR-10
            self.model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True
            )
            # The loaded model already includes the final classification layer for 10 classes
        except Exception as e:
            print(f"Error loading model from torch hub: {e}")
            print(
                "Please ensure internet connection and correct hub syntax/repository availability."
            )
            raise

    def forward(self, x):
        return self.model(x)


# --- Load Model ---
try:
    model = CIFAR10Model()
    model.to(device)  # Move model to the appropriate device
    model.eval()  # Set the model to evaluation mode (important!)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to initialize or load the model. Exiting. Error: {e}")
    exit()

# --- CIFAR-10 Preprocessing ---
# Normalization values commonly used for CIFAR-10 pre-trained models
# These might differ slightly depending on the specific training, but are standard.
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
)

# Transformation sequence: PIL Image -> Tensor -> Normalize
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),  # Converts PIL image (H x W x C) [0, 255] to Tensor (C x H x W) [0.0, 1.0]
        normalize,  # Normalizes Tensor with mean and std
    ]
)

# --- Load CIFAR-10 Data ---
# Download=True will download if not found in the root directory
try:
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True
    )
    print("CIFAR-10 test dataset loaded/downloaded.")
except Exception as e:
    print(f"Failed to load or download CIFAR-10 dataset. Error: {e}")
    exit()

# --- Get a Single Image ---
# Let's take the first image from the test set
image_index = 0
# The dataset returns (PIL Image, label)
pil_image, true_label_idx = test_dataset[image_index]
print(f"Selected image index: {image_index}")

# --- Prepare Image for Model ---
# Apply the preprocessing transforms
# The output is a Tensor (C x H x W)
input_tensor = preprocess(pil_image)

# Add a batch dimension (N=1): Model expects input shape (N x C x H x W)
# Use unsqueeze(0) to add dimension at the beginning
input_batch = input_tensor.unsqueeze(0)  # Shape: [1, 3, 32, 32]

# Move the input tensor to the same device as the model
input_batch = input_batch.to(device)

# --- Dimension Check ---
print("\n--- Dimension Check ---")
print("Original PIL Image size:", pil_image.size)  # (Width, Height)
print(
    "Input Tensor shape (after ToTensor):", input_tensor.shape
)  # [C, H, W] -> [3, 32, 32]
print(
    "Input Batch shape (after unsqueeze):", input_batch.shape
)  # [N, C, H, W] -> [1, 3, 32, 32]
print("Input Batch device:", input_batch.device)
print("-----------------------")


# --- Perform Prediction ---
with torch.no_grad():  # Disable gradient calculations for inference
    output = model(input_batch)  # Forward pass

# --- Process Output ---
# Output contains raw scores (logits) for each of the 10 classes
# Shape: [batch_size, num_classes] -> [1, 10]
print("\n--- Prediction ---")
print("Output logits shape:", output.shape)
# print("Output logits:", output)

# Get the index of the highest score (predicted class)
# We use dim=1 because dim=0 is the batch dimension
predicted_idx = torch.argmax(
    output, dim=1
).item()  # .item() gets the Python number from a 0-dim tensor

# --- Map Index to Class Name ---
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

predicted_class = classes[predicted_idx]
true_class = classes[true_label_idx]

print(f"True Label Index: {true_label_idx}, True Class: {true_class}")
print(f"Predicted Label Index: {predicted_idx}, Predicted Class: {predicted_class}")

# --- (Optional) Display the Image ---
plt.imshow(pil_image)  # Display the original PIL image
plt.title(f"True: {true_class}\nPredicted: {predicted_class}")
plt.axis("off")
plt.show()

# |%%--%%| <sEei6hocnX|aG1J5z88uZ>


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        embed_dim (int): Output embedding dimension of the patch.
        patch_size (int): Size of the patches to convert input image into.
        num_patches (int): The number of patches. (Used for positional embedding)
        dropout (float): Dropout rate.
        in_channels (int): Number of color channels for the input images.
    """

    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Create the Conv2d layer for patching
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,  # No padding needed if image size is divisible by patch size
        )

        # Create the flatten layer
        self.flatten = nn.Flatten(
            start_dim=2, end_dim=3  # only flatten the spatial dimensions
        )

        # Learnable class token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim),
            requires_grad=True,  # Shape: (batch_size=1, num_tokens=1, embed_dim)
        )

        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim),
            requires_grad=True,  # Add 1 for class token
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Check input shape
        assert (
            x.shape[2] == x.shape[3] and x.shape[2] % self.patch_size == 0
        ), f"Input image size ({x.shape[2]}x{x.shape[3]}) must be square and divisible by patch size ({self.patch_size})"

        # Get the batch size
        batch_size = x.shape[0]

        # Create patches
        x_patched = self.patcher(x)  # Shape: (batch_size, embed_dim, H/P, W/P)

        # Flatten the spatial dimensions
        x_flattened = self.flatten(
            x_patched
        )  # Shape: (batch_size, embed_dim, num_patches)

        # Permute the flattened patches to match Transformer expectation (batch_size, num_patches, embed_dim)
        x_permuted = x_flattened.permute(
            0, 2, 1
        )  # Shape: (batch_size, num_patches, embed_dim)

        # Expand the class token across the batch dimension
        cls_token_expanded = self.cls_token.expand(
            batch_size, -1, -1
        )  # Shape: (batch_size, 1, embed_dim)

        # Prepend the class token to the patch embeddings
        x = torch.cat(
            (cls_token_expanded, x_permuted), dim=1
        )  # Shape: (batch_size, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = self.position_embeddings + x

        # Apply dropout
        x = self.dropout(x)
        return x


# --- Test PatchEmbedding ---
print("--- Testing PatchEmbedding ---")
test_patch_embedding = PatchEmbedding(
    EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS
).to(device)
test_x = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
test_output = test_patch_embedding(test_x)
print(f"Input shape: {test_x.shape}")
print(
    f"Output shape (PatchEmbedding): {test_output.shape}"
)  # Expected: (BATCH_SIZE, NUM_PATCHES + 1, EMBED_DIM)
print("-" * 30)

# |%%--%%| <aG1J5z88uZ|27cLuxrlk1>


class ViT(nn.Module):
    """Creates a Vision Transformer architecture.

    Args:
      img_size (int): Height/width of input image (must be square). Default 224.
      in_channels (int): Number of channels in input image. Default 3.
      patch_size (int): Size of patch. Default 16.
      num_transformer_layers (int): Number of Transformer Encoder layers. Default 12.
      embedding_dim (int): Hidden size D from Table 1 for ViT-Base. Default 768.
      mlp_size (int): Size of the FeedForward layer. Default 3072.
      num_heads (int): Number of heads in Multi-Head Attention layer. Default 12.
      attn_dropout (float): Dropout for attention layers. Default 0.
      mlp_dropout (float): Dropout for feed-forward layers. Default 0.1.
      embedding_dropout (float): Dropout for patch and position embeddings. Default 0.1.
      num_classes (int): Number of classes to classify. Default 1000.
    """

    def __init__(
        self,
        num_patches,
        img_size,  # Keep img_size for potential validation inside PatchEmbedding
        num_classes,
        patch_size,
        embed_dim,
        num_encoders,
        num_heads,
        hidden_dim,  # Note: hidden_dim is often mlp_dim in TransformerEncoderLayer, not used explicitly here if using default nn.TransformerEncoderLayer feedforward dim
        dropout,  # Used for embedding dropout and transformer layer dropout
        activation,  # Used in TransformerEncoderLayer
        in_channels,
    ):
        super().__init__()

        # Check image size is divisible by patch size
        assert (
            img_size % patch_size == 0
        ), f"Image size ({img_size}) must be divisible by patch size ({patch_size})"

        # 1. Create Patch Embedding layer
        self.embeddings_block = PatchEmbedding(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_patches=num_patches,
            dropout=dropout,  # Use the main dropout for embedding dropout
            in_channels=in_channels,
        )

        # 2. Create Transformer Encoder blocks
        # Note: hidden_dim (often called mlp_dim) in nn.TransformerEncoderLayer defaults to d_model * 4
        # If you want to explicitly set it, you need to pass dim_feedforward=hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,  # Explicitly set the feedforward dimension
            dropout=dropout,  # This dropout applies to attention and feedforward within the layer
            activation=activation,
            batch_first=True,  # Input/output shape: (batch, seq, feature)
            norm_first=True,  # Pre-Normalization (LayerNorm before MHA/FFN) - often performs better
        )
        self.encoder_blocks = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoders
        )

        # 3. Create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x):
        # Pass input through Patch Embedding
        x = self.embeddings_block(x)  # Shape: (batch_size, num_patches + 1, embed_dim)

        # Pass patch embeddings through Transformer Encoder blocks
        x = self.encoder_blocks(x)  # Shape: (batch_size, num_patches + 1, embed_dim)

        # Get the CLS token embedding (it's the first one)
        cls_token_embedding = x[:, 0, :]  # Shape: (batch_size, embed_dim)

        # Pass CLS token embedding through MLP head
        y = self.mlp_head(cls_token_embedding)  # Shape: (batch_size, num_classes)
        return y


# --- Test ViT ---
print("--- Testing ViT Model ---")
model = ViT(
    NUM_PATCHES,
    IMG_SIZE,
    NUM_CLASSES,
    PATCH_SIZE,
    EMBED_DIM,
    NUM_ENCODERS,
    NUM_HEADS,
    HIDDEN_DIM,
    DROPOUT,
    ACTIVATION,
    IN_CHANNELS,
).to(device)

test_x_vit = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
test_output_vit = model(test_x_vit)
print(f"Input shape: {test_x_vit.shape}")
print(
    f"Output shape (ViT): {test_output_vit.shape}"
)  # Expected: (BATCH_SIZE, NUM_CLASSES)
# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")
print("-" * 30)

# |%%--%%| <27cLuxrlk1|PsXhzxYQlM>

# ---- Data Loading using torchvision.datasets.MNIST ----

# Define transforms
# Note: Using standard MNIST normalization values
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(15),  # Keep data augmentation for training
        transforms.ToTensor(),  # Converts PIL Image (0-255) to FloatTensor (0.0-1.0)
        transforms.Normalize((0.1307,), (0.3081,)),  # Mean and Std deviation for MNIST
    ]
)

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Download and load the datasets
train_dataset = MNIST(
    root=DATA_ROOT, train=True, download=True, transform=train_transform
)

# Use the official test set as the validation set for simplicity here
# For rigorous results, split the training set instead.
val_dataset = MNIST(
    root=DATA_ROOT,
    train=False,  # Use the test split
    download=True,
    transform=test_transform,  # Use test transform (no augmentation)
)

test_dataset = MNIST(
    root=DATA_ROOT, train=False, download=True, transform=test_transform
)

print(f"Size of Training Set: {len(train_dataset)}")
print(f"Size of Validation Set: {len(val_dataset)}")
print(f"Size of Test Set: {len(test_dataset)}")

# --- Display a sample image ---
img_display, label_display = train_dataset[0]
print(f"Image shape: {img_display.shape}, Label: {label_display}")
plt.figure()
# Need to un-normalize and permute dimensions for display if normalized
# Or display directly if ToTensor() was the last step (before normalization)
# Since we normalized, let's display the tensor directly (will look weird but shows structure)
# To display properly: img_display = img_display * 0.3081 + 0.1307
plt.imshow(img_display.squeeze(), cmap="gray")
plt.title(f"Sample Image - Label: {label_display}")
plt.show()

# |%%--%%| <PsXhzxYQlM|0N68HizUNm> - Cell marker retained, content above replaced custom datasets


# Create DataLoaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,  # Adjust based on your system
    pin_memory=True,  # Can speed up data transfer to GPU
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # No need to shuffle validation data
    num_workers=2,
    pin_memory=True,
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # No need to shuffle test data
    num_workers=2,
    pin_memory=True,
)

# Check a batch
images, labels = next(iter(train_dataloader))
print(
    f"Batch image shape: {images.shape}"
)  # Should be [BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE]
print(f"Batch labels shape: {labels.shape}")  # Should be [BATCH_SIZE]

# |%%--%%| <0N68HizUNm|mzwqtJPZm3>

# --- Training Setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    betas=ADAM_BETAS,
    lr=LEARNING_RATE,
    weight_decay=ADAM_WEIGHT_DECAY,
)

# Lists to store metrics
train_losses = []
val_losses = []
train_accs = []
val_accs = []

print("\n--- Starting Training ---")
start = timeit.default_timer()

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    train_labels_epoch = []
    train_preds_epoch = []
    train_running_loss = 0.0
    # Use tqdm for the dataloader directly
    for idx, (img, label) in enumerate(
        tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    ):
        # Move data to device
        # img shape is already (batch, channel, height, width) from ToTensor()
        img = img.float().to(device)
        # label shape is (batch), ensure it's Long type for CrossEntropyLoss
        label = label.long().to(device)

        # Forward pass
        y_pred = model(img)  # Output shape: (batch, num_classes)

        # Calculate loss
        loss = criterion(y_pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_running_loss += loss.item()

        # Store predictions and labels for accuracy calculation
        y_pred_label = torch.argmax(y_pred, dim=1)
        train_labels_epoch.extend(label.cpu().numpy())
        train_preds_epoch.extend(y_pred_label.cpu().numpy())

    train_loss = train_running_loss / len(train_dataloader)
    train_accuracy = sum(
        1 for x, y in zip(train_preds_epoch, train_labels_epoch) if x == y
    ) / len(train_labels_epoch)
    train_losses.append(train_loss)
    train_accs.append(train_accuracy)

    # --- Validation Phase ---
    model.eval()
    val_labels_epoch = []
    val_preds_epoch = []
    val_running_loss = 0.0
    with torch.no_grad():
        for idx, (img, label) in enumerate(
            tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validate]", leave=False
            )
        ):
            img = img.float().to(device)
            label = label.long().to(device)

            # Forward pass
            y_pred = model(img)

            # Calculate loss
            loss = criterion(y_pred, label)
            val_running_loss += loss.item()

            # Store predictions and labels
            y_pred_label = torch.argmax(y_pred, dim=1)
            val_labels_epoch.extend(label.cpu().numpy())
            val_preds_epoch.extend(y_pred_label.cpu().numpy())

    val_loss = val_running_loss / len(val_dataloader)
    val_accuracy = sum(
        1 for x, y in zip(val_preds_epoch, val_labels_epoch) if x == y
    ) / len(val_labels_epoch)
    val_losses.append(val_loss)
    val_accs.append(val_accuracy)

    # Print epoch results
    print("-" * 30)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"  Valid Loss: {val_loss:.4f} | Valid Accuracy: {val_accuracy:.4f}")
    print("-" * 30)

stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")
print("--- Training Finished ---")

# |%%--%%| <mzwqtJPZm3|iocutqjw2B>

# Plotting training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Validation Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), train_accs, label="Train Accuracy")
plt.plot(range(1, EPOCHS + 1), val_accs, label="Validation Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# |%%--%%| <iocutqjw2B|123mLLQium>

# --- Evaluation on Test Set ---
print("\n--- Evaluating on Test Set ---")
model.eval()
test_labels = []
test_preds = []
test_running_loss = 0.0

with torch.no_grad():
    for idx, (img, label) in enumerate(
        tqdm(test_dataloader, desc="Testing", leave=False)
    ):
        img = img.float().to(device)
        label = label.long().to(device)

        # Forward pass
        outputs = model(img)
        loss = criterion(outputs, label)
        test_running_loss += loss.item()

        # Get predictions
        y_pred_label = torch.argmax(outputs, dim=1)
        test_labels.extend(label.cpu().numpy())
        test_preds.extend(y_pred_label.cpu().numpy())

test_loss = test_running_loss / len(test_dataloader)
test_accuracy = sum(1 for x, y in zip(test_preds, test_labels) if x == y) / len(
    test_labels
)

print("-" * 30)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("-" * 30)

# Clean up GPU memory if needed
# torch.cuda.empty_cache() # Usually not necessary here unless running interactively and memory is tight

# |%%--%%| <123mLLQium|xKlZxZnRRp>

# Optional: Show some predictions from the test set
print("\n--- Sample Test Predictions ---")
num_samples_to_show = 6
indices = random.sample(range(len(test_dataset)), num_samples_to_show)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(indices):
    img, true_label = test_dataset[idx]  # Get original image and label
    model_input = img.unsqueeze(0).to(device)  # Add batch dimension and move to device

    model.eval()
    with torch.no_grad():
        pred_logit = model(model_input)
        pred_label = torch.argmax(pred_logit, dim=1).item()

    plt.subplot(2, 3, i + 1)
    # Unnormalize for display if needed, or show raw tensor
    # img_display = img.squeeze() * 0.3081 + 0.1307 # Unnormalize
    img_display = img.squeeze()  # Show normalized
    plt.imshow(img_display.cpu(), cmap="gray")
    plt.title(
        f"True: {true_label} | Pred: {pred_label}",
        color=("green" if true_label == pred_label else "red"),
    )
    plt.axis("off")

plt.tight_layout()
plt.show()


# |%%--%%| <xKlZxZnRRp|OVI8TgBle3>

# --- Remove Submission Code ---
# The following code related to creating submission.csv is removed
# as we are now evaluating on the standard MNIST test set which has labels.

# submission_df = pd.DataFrame(list(zip(ids, labels)), columns=["ImageId", "Label"])
# submission_df.to_csv("submission.csv", index=False)
# submission_df.head()
print(
    "\n Evaluation complete. No submission file generated as standard MNIST test set was used."
)

# --- END OF FILE vit_implementation.py ---
