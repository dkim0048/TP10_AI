import io
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

MODEL_PATH = (
    Path(__file__).parent.parent
    / "photo_mission_image_classification"
    / "cnn"
    / "model"
    / "best_model.pth"
)

CATEGORY_MAP = {
    "cat": "animal",
    "dog": "animal",
    "bird": "animal",
    "flower": "nature",
    "fruit": "nature",
    "vegetable": "nature",
    "car": "daily_life",
    "bicycle": "daily_life",
    "shoe": "daily_life",
}

IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


try:
    _checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    CLASS_NAMES: list[str] = _checkpoint["class_names"]

    _cnn = BaselineCNN(num_classes=len(CLASS_NAMES))
    _cnn.load_state_dict(_checkpoint["model_state"])
    _cnn.eval()

except FileNotFoundError:
    raise RuntimeError(f"CNN model file not found: {MODEL_PATH}")
except (KeyError, RuntimeError) as e:
    raise RuntimeError(f"Failed to load CNN model: {e}")


_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

router = APIRouter(prefix="/image")


class ClassifyResponse(BaseModel):
    predicted_class: str
    category: str
    confidence: float


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    tensor = _transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = _cnn(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(1)

    predicted_class = CLASS_NAMES[pred_idx.item()]
    category = CATEGORY_MAP.get(predicted_class, "unknown")

    return ClassifyResponse(
        predicted_class=predicted_class,
        category=category,
        confidence=round(confidence.item(), 4),
    )