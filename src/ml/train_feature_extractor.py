import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import json
import matplotlib.pyplot as plt

class CustomLoss(nn.Module):
    def __init__(self, property_start_idx):
        super(CustomLoss, self).__init__()
        self.property_start_idx = property_start_idx
        
    def forward(self, reconstructed, original, latent):
        # Genel rekonstrüksiyon kaybı
        reconstruction_loss = F.mse_loss(reconstructed, original)
        
        # Özellik tahminleri için ağırlıklı kayıp
        prop_features = original[:, self.property_start_idx:]
        prop_predictions = reconstructed[:, self.property_start_idx:]
        
        # E_coh için daha yüksek ağırlık (ilk özellik)
        e_coh_loss = 2.0 * F.mse_loss(
            prop_predictions[:, 0:1], 
            prop_features[:, 0:1]
        )
        
        # Diğer özellikler için normal ağırlık
        other_prop_loss = F.mse_loss(
            prop_predictions[:, 1:], 
            prop_features[:, 1:]
        )
        
        # Latent space düzenlileştirme
        latent_regularization = 0.01 * torch.mean(torch.abs(latent))
        
        total_loss = reconstruction_loss + e_coh_loss + other_prop_loss + latent_regularization
        return total_loss

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class MXeneDataset(Dataset):
    def __init__(self, comp_features, elem_prop_features, prop_features):
        # Combine all features
        self.features = np.concatenate(
            [comp_features, elem_prop_features, prop_features], 
            axis=1
        )
        self.features = torch.FloatTensor(self.features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=300, device='cpu', property_start_idx=None):
    """Train the autoencoder"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            reconstructed, latent = model(batch)
            loss = criterion(reconstructed, batch, latent)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed, latent = model(batch)
                loss = criterion(reconstructed, batch, latent)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping ve model kaydetme
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(model_dir, 'feature_extractor_best.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def plot_training_history(train_losses, val_losses, save_path):
    """Plot and save training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models', 'feature_extractor')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Load data
    comp_features = np.load(os.path.join(data_dir, 'composition_features.npy'))
    elem_prop_features = np.load(os.path.join(data_dir, 'element_prop_features.npy'))
    prop_features = np.load(os.path.join(data_dir, 'property_features.npy'))
    
    # Load feature information
    with open(os.path.join(data_dir, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    # Calculate property start index
    property_start_idx = (feature_info['composition_size'] + 
                         feature_info['element_prop_size'])
    
    # Prepare data
    dataset = MXeneDataset(comp_features, elem_prop_features, prop_features)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model and training components
    input_dim = (feature_info['composition_size'] + 
                feature_info['element_prop_size'] + 
                feature_info['property_size'])
    
    model = Autoencoder(input_dim, latent_dim=16)
    criterion = CustomLoss(property_start_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=300, device=device, property_start_idx=property_start_idx
    )
    
    # Plot and save training history
    plot_training_history(
        train_losses, val_losses,
        os.path.join(model_dir, 'training_history.png')
    )
