import torch.nn as nn

# Definir la Red Neuronal
class ModeloNN(nn.Module):
    def __init__(self, input_size):
        
        super(ModeloNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)  # Probabilidad de desactivación de 50%

    def forward(self, x):
        x = self.relu(self.fc1(x))             # Capa 1 + ReLU
        x = self.dropout(x)                    # Dropout después de la primera capa
        x = self.relu(self.fc2(x))             # Capa 2 + ReLU
        x = self.dropout(x)                    # Dropout después de la segunda capa
        x = self.relu(self.fc3(x))             # Capa 3 + ReLU
        x = self.fc4(x)                        # Capa de salida (sin activación ReLU ya que usamos Sigmoid más tarde)
        x = self.sigmoid(x)                    # Aplicar Sigmoid para obtener probabilidad
        return x