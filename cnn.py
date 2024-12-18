import os
import cv2
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

train_data_root = './chest_xray/train'
test_data_root = './chest_xray/test'
class_folders = [folder for folder in os.listdir(train_data_root) if os.path.isdir(os.path.join(train_data_root, folder))]
class_to_label = {class_name: label for label, class_name in enumerate(class_folders)}

# transform = v2.Compose([
#     v2.ToDtype(torch.float32, scale = True),
#     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

def data_labels(data_root):
    data = []
    labels = []
    for class_name in class_folders:
        class_folder = os.path.join(data_root, class_name)
        class_label = class_to_label[class_name]

        for image_filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Resize all images to a consistent size (224x224)
            image = cv2.resize(image, (224, 224))

            #image = transform(image)
            data.append(torch.from_numpy(image).unsqueeze(0).float())
            labels.append(torch.tensor(class_label))

    data = torch.stack(data)
    labels = torch.stack(labels)  # Corrected to stack the labels

    return data, labels


train_data, train_labels = data_labels(train_data_root)
test_data, test_labels = data_labels(test_data_root)

train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

batch_size = 1

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        #self.conv2 = nn.Conv2d(32, 64, 3) 
        self.pool = nn.MaxPool2d(3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 74 * 74, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.pool(x)
        x = self.flatten(x)  # Reshape for the fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNN_with_Autoencoder(nn.Module):
    def __init__(self, autoencoder):
        super(CNN_with_Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(32*18*18,128)
        self.fc2 = nn.Linear(128,2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.autoencoder = autoencoder
    
    def forward(self, x):
        x = self.autoencoder.encoder(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
model = CNN_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.00075)
num_epochs = 10
train_loss = []

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# autoencoder = Autoencoder()
# criterion_autoencoder = nn.MSELoss()
# optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr = 0.0001)
# num_epochs = 20

# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         inputs, _ = batch
#         optimizer_autoencoder.zero_grad()
#         outputs = autoencoder(inputs)
#         loss_autoencoder = criterion_autoencoder(outputs, inputs)
#         loss_autoencoder.backward()
#         optimizer_autoencoder.step()
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_autoencoder.item():.4f}')

# cnn_autoencoder = CNN_with_Autoencoder(autoencoder)
# criterion_cnn = nn.CrossEntropyLoss()
# optimizer_cnn = optim.Adam(cnn_autoencoder.parameters(), lr = 0.0005)

# num_epochs = 25
# train_loss = []

# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         inputs, labels = batch
#         optimizer_cnn.zero_grad()
#         outputs = cnn_autoencoder(inputs)
#         loss_cnn = criterion_cnn(outputs, labels)
#         loss_cnn.backward()
#         optimizer_cnn.step()
#     train_loss.append(loss_cnn.item())
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_cnn.item():.4f}')
    
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
epochs = (range(1, num_epochs+1))
plt.plot(epochs, train_loss,'b',label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()