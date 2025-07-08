import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score

def validate_model(model, criterion, dataloader, device):
    model.eval()  
    total_loss = 0.0  
    total_samples = 0
    all_predictions = []  
    all_targets = []  

    with torch.no_grad():  
        for inputs, targets in dataloader:  
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs)  
            loss = criterion(outputs, targets)  
            total_loss += loss.item() * inputs.size(0)  

            _, predicted = torch.max(outputs, 1) 
            all_predictions.extend(predicted.cpu().numpy())  
            all_targets.extend(targets.cpu().numpy()) 
            total_samples += targets.size(0)  

    avg_loss = total_loss / total_samples  
    accuracy = accuracy_score(all_targets, all_predictions)  

    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0) 
    recall = recall_score(all_targets, all_predictions, average='macro')  

    return avg_loss, accuracy, precision, recall 