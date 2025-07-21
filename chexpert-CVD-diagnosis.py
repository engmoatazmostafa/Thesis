import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Configuration ---
# Set the path to your CheXpert-v1.0-small directory
data_dir = './chexpert_dataset/CheXpert-v1.0-small'
train_csv_path = os.path.join(data_dir, 'train.csv')
valid_csv_path = os.path.join(data_dir, 'valid.csv')

# Labels of interest for CheXpert. We'll focus on 'Cardiomegaly' for CVD.
# Other relevant labels might include 'Edema', 'Enlarged Cardiomediastinum'
# For a comprehensive CVD diagnosis, you might combine several of these.
CVD_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

# Index of 'Cardiomegaly' in the CVD_LABELS list
CARDIOMEGALY_IDX = CVD_LABELS.index('Cardiomegaly')

# Image dimensions
IMG_SIZE = 320
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, uncertainty_strategy='U-Ones'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
            uncertainty_strategy (string): How to handle uncertain labels (-1).
                                           'U-Ones': Treat -1 as 1 (positive).
                                           'U-Zeros': Treat -1 as 0 (negative).
                                           'U-Ignore': Ignore samples with -1 (not recommended for training).
        """
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.uncertainty_strategy = uncertainty_strategy

        # Map original CheXpert image paths to full paths
        # The 'Path' column in CSV is relative to CheXpert-v1.0-small
        self.dataframe['Path'] = self.dataframe['Path'].apply(
            lambda x: os.path.join(self.root_dir, x)
        )

        # Select only the relevant labels for training
        self.labels = self.dataframe[CVD_LABELS].values

        # Apply uncertainty strategy
        # CheXpert labels: 1 (positive), 0 (negative), -1 (uncertain), NaN (unmentioned)
        # Convert NaN to 0 (unmentioned means negative for this task)
        self.labels[np.isnan(self.labels)] = 0

        if self.uncertainty_strategy == 'U-Ones':
            self.labels[self.labels == -1] = 1
        elif self.uncertainty_strategy == 'U-Zeros':
            self.labels[self.labels == -1] = 0
        elif self.uncertainty_strategy == 'U-Ignore':
            # This strategy is more complex as it requires filtering rows.
            # For simplicity, we'll keep it as U-Ones/U-Zeros for this example.
            # If you truly want to ignore, you'd filter the dataframe here.
            print("Warning: 'U-Ignore' strategy for uncertainty not fully implemented for training in this example. Using U-Ones/U-Zeros instead.")
            self.labels[self.labels == -1] = 1 # Default to U-Ones if U-Ignore is selected without full implementation.


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.iloc[idx]['Path']
        image = Image.open(img_path).convert('RGB') # Ensure 3 channels

        labels = self.labels[idx].astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(labels)

# --- Transformations ---
# Define transformations for training and validation data
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load Data ---
print(f"Loading data from: {data_dir}")
train_dataset = CheXpertDataset(csv_file=train_csv_path,
                                root_dir=data_dir,
                                transform=train_transforms,
                                uncertainty_strategy='U-Ones') # Common strategy for CheXpert
val_dataset = CheXpertDataset(csv_file=valid_csv_path,
                              root_dir=data_dir,
                              transform=val_transforms,
                              uncertainty_strategy='U-Ones')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# --- Model Definition ---
# Using a pre-trained DenseNet121 as a feature extractor
class CheXpertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CheXpertClassifier, self).__init__()
        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        # Replace the classifier layer
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.densenet(x)

model = CheXpertClassifier(num_classes=len(CVD_LABELS)).to(DEVICE)

# --- Loss Function and Optimizer ---
# For multi-label classification, Binary Cross-Entropy with Logits is suitable
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                # Apply sigmoid to get probabilities for AUC calculation
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {val_epoch_loss:.4f}")

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Calculate AUC for all labels
        try:
            aucs = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in range(len(CVD_LABELS))]
            mean_auc = np.mean(aucs)
            print(f"Mean AUC over all {len(CVD_LABELS)} labels: {mean_auc:.4f}")
            print(f"AUC for Cardiomegaly: {aucs[CARDIOMEGALY_IDX]:.4f}")
        except ValueError as e:
            print(f"Could not calculate AUC: {e}. This might happen if a class has only one label type in the batch.")

        scheduler.step(val_epoch_loss)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            # Save the best model
            torch.save(model.state_dict(), 'best_chexpert_model.pth')
            print("Model saved as 'best_chexpert_model.pth'")

        end_time = time.time()
        print(f"Epoch {epoch+1} finished in {(end_time - start_time):.2f} seconds.")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS)

    print("\n--- Training Complete ---")
    print("Loading the best model for final evaluation...")
    model.load_state_dict(torch.load('best_chexpert_model.pth'))
    model.eval()

    # --- Final Evaluation (Optional, on validation set for demonstration) ---
    # For a real scenario, you'd use a separate test set.
    final_preds = []
    final_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            final_preds.append(preds)
            final_labels.append(labels.cpu().numpy())

    final_preds = np.vstack(final_preds)
    final_labels = np.vstack(final_labels)

    # Convert probabilities to binary predictions (e.g., using a threshold of 0.5)
    # This is for metrics like precision, recall, F1-score. For AUC, probabilities are used.
    threshold = 0.5
    binary_preds = (final_preds > threshold).astype(int)

    print("\n--- Final Evaluation on Validation Set ---")
    for i, label_name in enumerate(CVD_LABELS):
        print(f"\n--- Metrics for {label_name} ---")
        try:
            auc = roc_auc_score(final_labels[:, i], final_preds[:, i])
            print(f"AUC: {auc:.4f}")
            # Classification report for precision, recall, f1-score
            report = classification_report(final_labels[:, i], binary_preds[:, i], zero_division=0)
            print(report)
            # Confusion Matrix
            cm = confusion_matrix(final_labels[:, i], binary_preds[:, i])
            print("Confusion Matrix:")
            print(cm)
        except Exception as e:
            print(f"Could not calculate metrics for {label_name}: {e}")

    # Example: Visualize a prediction (requires a function to de-normalize and display)
    # This part is illustrative and not fully implemented for brevity.
    # def visualize_prediction(image_tensor, true_labels, predicted_probs, idx):
    #     # Denormalize image
    #     inv_normalize = transforms.Normalize(
    #         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    #         std=[1/0.229, 1/0.224, 1/0.225]
    #     )
    #     img = inv_normalize(image_tensor).permute(1, 2, 0).cpu().numpy()
    #     img = np.clip(img, 0, 1) # Clip values to [0,1] for display
    #
    #     plt.imshow(img)
    #     plt.title(f"True: {CVD_LABELS[np.where(true_labels == 1)[0]]}\nPred: {CVD_LABELS[np.argmax(predicted_probs)]} ({np.max(predicted_probs):.2f})")
    #     plt.axis('off')
    #     plt.show()

    # You could call visualize_prediction with a sample from your validation set
    # For example:
    # sample_image, sample_labels = val_dataset[0]
    # sample_image_tensor = sample_image.unsqueeze(0).to(DEVICE)
    # with torch.no_grad():
    #     sample_output = model(sample_image_tensor)
    #     sample_probs = torch.sigmoid(sample_output).cpu().numpy()[0]
    # visualize_prediction(sample_image, sample_labels, sample_probs, CARDIOMEGALY_IDX)
