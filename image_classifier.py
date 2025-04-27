import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class IrisImageClassifier(nn.Module):
    def __init__(self):
        super(IrisImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 3)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ImagePredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IrisImageClassifier().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        self.classes = ['Setosa', 'Versicolor', 'Virginica']

    def load_model(self, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image):
        try:
            with torch.no_grad():
                input_tensor = self.preprocess_image(image)
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                return {
                    'class': self.classes[predicted.item()],
                    'confidence': confidence.item() * 100
                }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None