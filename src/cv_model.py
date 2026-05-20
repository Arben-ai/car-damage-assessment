import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import json
import numpy as np
import cv2
from pathlib import Path

DAMAGE_CLASSES = ['broken_glass', 'broken_lights', 'dents', 'lost_parts', 'punctured', 'scratch', 'torn']

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


def compute_gradcam(image: Image.Image, model: nn.Module, class_idx: int,
                    device: str = 'cpu') -> Image.Image:
    """
    Runs GradCAM on the image using the last conv layer of EfficientNet.
    Returns a PIL image with heatmap overlay and a bounding box drawn
    around the highest-activation (damage) region.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    img_rgb  = image.convert('RGB')
    orig_w, orig_h = img_rgb.size

    tensor  = TRANSFORM(img_rgb).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(class_idx)]

    # conv_head is the last Conv2d before GlobalAvgPool in timm EfficientNet
    with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # (224, 224)

    # Heatmap overlay on 224x224 version of the image
    img_224      = np.array(img_rgb.resize((224, 224)), dtype=np.float32) / 255.0
    overlay_224  = show_cam_on_image(img_224, grayscale_cam, use_rgb=True)  # uint8 (224,224,3)

    # Find bounding box by thresholding the activation map
    mask      = (grayscale_cam > 0.45).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        # Draw box in 224x224 space (used for resize later)
        cv2.rectangle(overlay_224, (x, y), (x + w, y + h), (220, 38, 38), 2)

        # Scale box to original image coordinates for the label
        sx, sy   = orig_w / 224, orig_h / 224
        ox, oy   = int(x * sx), int(y * sy)
        ow, oh   = int(w * sx), int(h * sy)

        # Resize overlay to original image size, then add label at correct position
        overlay_orig = np.array(
            Image.fromarray(overlay_224).resize((orig_w, orig_h), Image.LANCZOS)
        )
        cv2.rectangle(overlay_orig, (ox, oy), (ox + ow, oy + oh), (220, 38, 38), 3)
        label_y = max(oy - 10, 20)
        cv2.putText(overlay_orig, 'Damage Region', (ox, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 38, 38), 2, cv2.LINE_AA)
        return Image.fromarray(overlay_orig)

    # No clear region found — just return the heatmap without a box
    return Image.fromarray(overlay_224).resize((orig_w, orig_h), Image.LANCZOS)
