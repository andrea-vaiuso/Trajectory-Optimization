import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import time
import torch

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, X_train=None):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 13)
        self.fc2 = nn.Linear(13, 40)
        self.fc3 = nn.Linear(40, output_dim)
        self.scaler = MinMaxScaler()
        if X_train is not None:
            self.scaler.fit(X_train)
    
    def forward(self, x):
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train_model(self, X_train, y_train, X_val, y_val, num_epochs, patience, lr=0.015, verbose=True):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        patience_counter = 0
        earlystopped = False
        train_losses = []
        val_losses = []
        best_epoch = 0

        if verbose: print("Normalizing data...")
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        if verbose: print("Converting data to tensors...")
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = criterion(val_outputs, y_val)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = self.state_dict()
                best_epoch = epoch + 1
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                earlystopped = True
                break
            
            if verbose and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}', end='\r')
        if verbose: print()
        if earlystopped and verbose: print(f'Early stopping at epoch {epoch+1} with best epoch: {best_epoch}')
        end_time = time.time()
        total_training_time = end_time - start_time
        print(f'Training completed in {total_training_time:.2f} seconds | Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}')
        self.load_state_dict(best_model)
        return train_losses, val_losses, best_epoch, total_training_time
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_normalized = self.scaler.transform(X)
            outputs = self(torch.tensor(X_normalized, dtype=torch.float32))
        return outputs.numpy()