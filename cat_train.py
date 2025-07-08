from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 
from torch.optim import Adam  
from torch.nn import CrossEntropyLoss  
from sklearn.metrics import confusion_matrix 

import torch 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  
import seaborn as sns 
import time  

from CatBreedModel import CatBreedModel 
from validation import * 

from PIL import ImageFile 

ImageFile.LOAD_TRUNCATED_IMAGES = True

train_transform = transforms.Compose([
    transforms.Resize((312, 312)),  
    transforms.RandomPerspective(0.2),  
    transforms.RandomVerticalFlip(0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5430, 0.4982, 0.4583], std=[0.2296, 0.2272, 0.2308])
])

test_transform = transforms.Compose([
    transforms.Resize((312, 312)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5430, 0.4982, 0.4583], std=[0.2296, 0.2272, 0.2308])
])

train_directory = "train_images/"
test_directory = "test_images/"  

batch_size = 4

train_dataset = datasets.ImageFolder(root=train_directory, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_directory, transform=test_transform)

data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.class_to_idx

print(train_dataset.class_to_idx)

model = CatBreedModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

learning_rate = 0.00005

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

num_epochs = 55

best_model_validation_accuracy = 0.0  

print("Starting to train...")

print(f"Dropout: 0.2 - Data Augmentation - Batch Size: {str(batch_size)} - Learning Rate: {str(learning_rate)} - Num Epochs: {str(num_epochs)} - Adam Optimizer - Cross Entropy")

start_time = time.time()

for epoch in range(num_epochs):  
    running_loss = 0.0  
    total_correct = 0
    total_samples = 0
    model.train()  

    for inputs, targets in data_loader:  
        inputs, targets = inputs.to(device), targets.to(device)  

        optimizer.zero_grad()  
        outputs = model(inputs) 
        loss = criterion(outputs, targets)  

        loss.backward()  
        optimizer.step() 

        _, predicted = torch.max(outputs, 1)  
        total_correct += (predicted == targets).sum().item() 
        total_samples += targets.size(0)  

        running_loss += loss.item() * inputs.size(0)  

    train_loss = running_loss / len(data_loader.dataset)

    val_loss, val_accuracy, prec, recall = validate_model(model, criterion, test_loader, device)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
        f'Train Loss: {train_loss:.4f}, '
        f'Val Loss: {val_loss:.4f}, '
        f'Val Acc: {val_accuracy:.4f}, '
        f'Model Acc: {total_correct/total_samples:.4f}, '
        f'Precision: {prec:.4f}, '
        f'Recall: {recall:.4f}')

    if val_accuracy > 0.86:
        best_model_validation_accuracy = val_accuracy

        torch.save(model.state_dict(), f"Model_22class_{val_accuracy*100:.2f}(DR0.2_DA)_Epoch_{str(epoch)}.pth")

end_time = time.time()
training_duration = end_time - start_time

print(f"Finished Training. Training duration: {training_duration:.2f} seconds")
print("Saving the model...")

torch.save(model.state_dict(), f"Model_22class_{val_accuracy*100:.2f}(DR0.2_DA)_NoEpoch_{str(epoch)}f.pth")