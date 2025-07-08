from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from CatBreedModel import CatBreedModel

model = CatBreedModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model89.pth", map_location=device))
model.eval()

image_path = "cat.jpg"
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize((312, 312)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5408, 0.4967, 0.4574], std=[0.2283, 0.2262, 0.2303])
])

image_tensor = preprocess(image)
image_tensor = image_tensor.unsqueeze(0)

class_dict = {'ABY': 0, 'ACL': 1, 'BEN': 2, 'BML': 3, 'BSH': 4, 'BUR': 5, 'CHA': 6,
                'CRX': 7, 'EUR': 8, 'EXO': 9, 'MAU': 10, 'MCO': 11, 'NFO': 12, 'OCI': 13,
                'OSH': 14, 'PER': 15, 'RAG': 16, 'RUS': 17, 'SBI': 18, 'SIA': 19, 'SOM': 20,
                'SPH': 21, 'TUV': 22}

with torch.no_grad():
    output = model(image_tensor)
    
    predicted_probs = nn.functional.softmax(output, dim=1)
    
    top_probs, top_indices = torch.topk(predicted_probs, 3)
    
    for i in range(3):
        prob = top_probs[0][i].item()
        index = top_indices[0][i].item()
        
        predicted_class = [key for key, value in class_dict.items() if value == index][0]

        print(f"Predicted class: {predicted_class}, Probability: {prob}")
        
predicted_index = torch.argmax(output, dim=1).item()

predicted_class = [key for key, value in class_dict.items() if value == predicted_index][0]