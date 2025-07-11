import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Önceden eğitilmiş ResNet50 modelini yükle
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Sınıflandırma katmanını (fc) atıp, özellik vektörünü almak için modeli ayarla
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval() # Modeli çıkarım (inference) moduna al

        # Görselleri modele uygun formata getirmek için transformasyonlar
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, img: Image.Image) -> np.ndarray:
        """
        Verilen bir PIL Image nesnesinden özellik vektörünü çıkarır.
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_t = self.transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            features = self.model(batch_t)
            # (1, 2048, 1, 1) şeklindeki tensörü (2048,) şeklinde bir numpy dizisine çevir
            squeezed_features = features.squeeze().numpy()
        
        return squeezed_features / np.linalg.norm(squeezed_features)

# Test için: extractor = FeatureExtractor()