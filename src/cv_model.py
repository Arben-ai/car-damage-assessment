import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import json
from pathlib import Path

DAMAGE_CLASSES = ['dents', 'scratch', 'broken_glass', 'broken_lights', 'lost_parts', 'torn', 'punctured', 'non_damaged']

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path: str, device: str = 'cpu') -> nn.Module:
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(DAMAGE_CLASSES))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict(image: Image.Image, model: nn.Module, device: str = 'cpu') -> dict:
    tensor = TRANSFORM(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    return {
        'damage_class': DAMAGE_CLASSES[pred_idx],
        'confidence': float(probs[pred_idx]),
        'class_probs': {cls: float(p) for cls, p in zip(DAMAGE_CLASSES, probs)}
    }
