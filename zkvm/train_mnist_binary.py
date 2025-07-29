# improved_mnist_training.py
import torch
import torch.nn as nn
import torchvision
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import os

# Kreiranje direktorijuma za čuvanje
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./weights', exist_ok=True)

# 1. Poboljšan Dataset preprocessing
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalizacija
    torchvision.transforms.Lambda(lambda x: (x > 0.0).float())  # binarizacija nakon normalizacije
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    torchvision.transforms.Lambda(lambda x: (x > 0.0).float())
])

# Učitavanje podataka
train_set = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_train, download=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_test, download=True)

# Train/Validation split
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_set)}")

# 2. Model sa 2 layera (da odgovara guest kodu)
class ImprovedMLP(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 64),       # Prvi layer: 784 -> 64
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 10)         # Output layer: 64 -> 10
        )
    
    def forward(self, x):
        return self.net(x)

# Kreiranje modela
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = ImprovedMLP(dropout_rate=0.3).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

print("Model architecture:")
print(model)

# Računanje broja parametara
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 3. Funkcija za testiranje
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss, all_preds, all_labels

# 4. Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# 5. Poboljšan trening sa validation i early stopping
num_epochs = 20
best_accuracy = 0.0
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# Liste za praćenje metrika
train_losses = []
val_losses = []
val_accuracies = []

print("Starting training...")
print("-" * 50)

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        if i % 200 == 199:  # Print every 200 batches
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/200:.4f}')
            running_loss = 0.0
    
    # Validation
    val_accuracy, val_loss, _, _ = evaluate_model(model, val_loader, device)
    train_loss = running_loss / num_batches if num_batches > 0 else 0
    
    # Dodavanje u liste
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    
    # Čuvanje najboljeg modela
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_accuracy,
            'val_loss': val_loss
        }, './models/best_mnist_model.pth')
        print(f'  *** New best accuracy: {best_accuracy:.4f} ***')
    
    # Early stopping
    if early_stopping(val_loss):
        print(f'Early stopping triggered at epoch {epoch+1}')
        break
    
    scheduler.step()
    print("-" * 50)

print(f'Training completed! Best validation accuracy: {best_accuracy:.4f}')

# 6. Učitavanje najboljeg modela i finalno testiranje
print("\nLoading best model for final evaluation...")
checkpoint = torch.load('./models/best_mnist_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

print("Final evaluation on test set:")
test_accuracy, test_loss, preds, labels = evaluate_model(model, test_loader, device)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Loss: {test_loss:.4f}')

print("\nDetailed Classification Report:")
print(classification_report(labels, preds, digits=4))

# 7. Export samo 2 layera (da odgovara guest kodu)
def to_fixed_improved(tensor, scale=1000):
    """Konvertuje tensor u fixed-point sa boljim skaliranjem"""
    # Clip ekstremne vrednosti
    tensor_clipped = torch.clamp(tensor, -10.0, 10.0)
    return (tensor_clipped.detach().cpu().numpy() * scale).astype(np.int32)

print("\nExporting model weights...")

# Export 2 layera
model.eval()
with torch.no_grad():
    # Layer 1: 784 -> 64
    W1 = model.net[1].weight.T  # Transponujemo za lakše korišćenje
    B1 = model.net[1].bias
    
    # Layer 2: 64 -> 10
    W2 = model.net[4].weight.T
    B2 = model.net[4].bias

# Proverite dimenzije
print(f"W1 shape: {W1.shape} (784 -> 64)")
print(f"B1 shape: {B1.shape} (64,)")
print(f"W2 shape: {W2.shape} (64 -> 10)")
print(f"B2 shape: {B2.shape} (10,)")

# Export u .npy format
np.save("./weights/W1.npy", to_fixed_improved(W1))
np.save("./weights/B1.npy", to_fixed_improved(B1))
np.save("./weights/W2.npy", to_fixed_improved(W2))
np.save("./weights/B2.npy", to_fixed_improved(B2))

print("All weights exported successfully to ./weights/ directory!")

# 8. Export u tekstualni format (za lakše korišćenje u C++)
print("\nExporting weights in text format...")
def export_to_text(tensor, filename):
    """Eksportuje tensor u tekstualni format"""
    with open(f"./weights/{filename}", 'w') as f:
        if tensor.ndim == 1:  # Bias
            f.write(f"{tensor.shape[0]}\n")
            for val in tensor:
                f.write(f"{int(val)}\n")
        else:  # Weight matrix
            f.write(f"{tensor.shape[0]} {tensor.shape[1]}\n")
            for i in range(tensor.shape[0]):
                for j in range(tensor.shape[1]):
                    f.write(f"{int(tensor[i,j])}")
                    if j < tensor.shape[1] - 1:
                        f.write(" ")
                f.write("\n")

export_to_text(to_fixed_improved(W1), "W1.txt")
export_to_text(to_fixed_improved(B1), "B1.txt")
export_to_text(to_fixed_improved(W2), "W2.txt")
export_to_text(to_fixed_improved(B2), "B2.txt")

print("Text format exports completed!")

# 9. Test sa jednim primerom
print("\nTesting with one sample:")
model.eval()
with torch.no_grad():
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    logits = model(x[:1])  # Prvi primer
    pred = torch.argmax(logits, dim=1)
    
    print(f"True label: {y[0].item()}")
    print(f"Predicted: {pred[0].item()}")
    print(f"Confidence: {torch.softmax(logits, dim=1)[0].max().item():.4f}")
    print(f"Logits: {logits[0].cpu().numpy()}")

# 10. Vizualizacija treninga (opcionalno)
try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', alpha=0.7)
    ax1.plot(val_losses, label='Validation Loss', alpha=0.7)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Training plots saved as './models/training_history.png'")
    
except ImportError:
    print("Matplotlib not available for plotting")

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Best Validation Accuracy: {best_accuracy:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Model saved to: ./models/best_mnist_model.pth")
print(f"Weights exported to: ./weights/")
print("="*60)