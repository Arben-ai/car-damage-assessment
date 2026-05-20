import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import timm
import json
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
    import numpy as np
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    img_rgb = image.convert('RGB')
    orig_w, orig_h = img_rgb.size

    tensor  = TRANSFORM(img_rgb).unsqueeze(0).to(device)
    targets = [ClassifierOutputTarget(class_idx)]

    # conv_head is the last Conv2d before GlobalAvgPool in timm EfficientNet
    with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # (224, 224)

    # Heatmap overlay on 224x224
    img_224     = np.array(img_rgb.resize((224, 224)), dtype=np.float32) / 255.0
    overlay_224 = show_cam_on_image(img_224, grayscale_cam, use_rgb=True)  # uint8 (224,224,3)

    # Resize overlay back to original size
    overlay_pil = Image.fromarray(overlay_224).resize((orig_w, orig_h), Image.LANCZOS)

    # Find bounding box of high-activation region using numpy (no cv2 needed)
    mask = grayscale_cam > 0.45
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        y_min, y_max = int(np.where(rows)[0][[0, -1]].tolist()[0]), int(np.where(rows)[0][[0, -1]].tolist()[1])
        x_min, x_max = int(np.where(cols)[0][[0, -1]].tolist()[0]), int(np.where(cols)[0][[0, -1]].tolist()[1])

        # Scale to original image size
        sx, sy = orig_w / 224, orig_h / 224
        ox1 = int(x_min * sx); oy1 = int(y_min * sy)
        ox2 = int(x_max * sx); oy2 = int(y_max * sy)

        draw = ImageDraw.Draw(overlay_pil)
        lw   = max(3, orig_w // 200)
        draw.rectangle([ox1, oy1, ox2, oy2], outline=(220, 38, 38), width=lw)
        # Label above the box
        label_y = max(oy1 - 24, 4)
        draw.rectangle([ox1, label_y, ox1 + 140, label_y + 20], fill=(220, 38, 38))
        draw.text((ox1 + 4, label_y + 2), 'Damage Region', fill=(255, 255, 255))

    return overlay_pil
