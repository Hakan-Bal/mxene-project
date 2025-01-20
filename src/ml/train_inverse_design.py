import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import json
from train_feature_extractor import Autoencoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class PropertyConditionedGenerator(nn.Module):
    def __init__(self, property_dim, latent_dim=16, hidden_dim=128):
        super(PropertyConditionedGenerator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(property_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, property_dim),
            nn.Sigmoid()  # Özellikler 0-1 arasında normalize edilmiş
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class InverseDesignDataset(Dataset):
    def __init__(self, properties, latent_features):
        self.properties = torch.FloatTensor(properties)
        self.latent_features = torch.FloatTensor(latent_features)
    
    def __len__(self):
        return len(self.properties)
    
    def __getitem__(self, idx):
        return self.properties[idx], self.latent_features[idx]

class CustomInverseLoss(nn.Module):
    def __init__(self):
        super(CustomInverseLoss, self).__init__()
    
    def forward(self, reconstructed_props, original_props, latent, target_latent):
        # Özellik rekonstrüksiyon kaybı
        prop_loss = F.mse_loss(reconstructed_props, original_props)
        
        # Latent space kaybı
        latent_loss = F.mse_loss(latent, target_latent)
        
        # Düzenlileştirme
        regularization = 0.01 * torch.mean(torch.abs(latent))
        
        return prop_loss + latent_loss + regularization

def train_inverse_model(model, train_loader, val_loader, criterion, optimizer, 
                       num_epochs=200, device='cpu'):
    """Train the inverse design model"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for props, latent_target in train_loader:
            props, latent_target = props.to(device), latent_target.to(device)
            optimizer.zero_grad()
            
            reconstructed_props, latent = model(props)
            loss = criterion(reconstructed_props, props, latent, latent_target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for props, latent_target in val_loader:
                props, latent_target = props.to(device), latent_target.to(device)
                reconstructed_props, latent = model(props)
                loss = criterion(reconstructed_props, props, latent, latent_target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping ve model kaydetme
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(model_dir, 'inverse_design_best.pth'))
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
    plt.title('Inverse Design Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models', 'inverse_design')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Load feature extractor model
    with open(os.path.join(data_dir, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    input_dim = (feature_info['composition_size'] + 
                feature_info['element_prop_size'] + 
                feature_info['property_size'])
    
    feature_extractor = Autoencoder(input_dim, 16)
    feature_extractor.load_state_dict(
        torch.load(os.path.join(base_dir, 'models', 'feature_extractor', 
                               'feature_extractor_best.pth'))
    )
    feature_extractor.eval()
    
    # Load data
    comp_features = np.load(os.path.join(data_dir, 'composition_features.npy'))
    elem_prop_features = np.load(os.path.join(data_dir, 'element_prop_features.npy'))
    prop_features = np.load(os.path.join(data_dir, 'property_features.npy'))
    
    # Generate latent representations
    combined_features = np.concatenate(
        [comp_features, elem_prop_features, prop_features], axis=1
    )
    with torch.no_grad():
        _, latent_features = feature_extractor(
            torch.FloatTensor(combined_features)
        )
        latent_features = latent_features.numpy()
    
    # Prepare data for inverse design
    property_scaler = StandardScaler()
    scaled_properties = property_scaler.fit_transform(prop_features)
    
    # Create datasets
    dataset = InverseDesignDataset(scaled_properties, latent_features)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize inverse design model
    inverse_model = PropertyConditionedGenerator(
        property_dim=prop_features.shape[1]
    )
    criterion = CustomInverseLoss()
    optimizer = torch.optim.AdamW(
        inverse_model.parameters(), 
        lr=0.001, 
        weight_decay=0.01
    )
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inverse_model = inverse_model.to(device)
    
    train_losses, val_losses = train_inverse_model(
        inverse_model, train_loader, val_loader, criterion, optimizer,
        num_epochs=200, device=device
    )
    
    # Plot and save training history
    plot_training_history(
        train_losses, val_losses,
        os.path.join(model_dir, 'inverse_design_training_history.png')
    )
    
    # Save property scaler
    import joblib
    joblib.dump(property_scaler, 
                os.path.join(model_dir, 'inverse_property_scaler.pkl'))
    
    print("Inverse design model training completed!")
