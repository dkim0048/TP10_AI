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
    "cat": "animals",
    "horse": "animals",
    "bird": "animals",
    "fish": "animals",
    "books": "daily_life",
    "chair": "daily_life",
    "umbrella": "daily_life",
    "airplane": "vehicles",
    "bicycle": "vehicles",
    "boat": "vehicles",
}

THEME_LABELS: dict[str, str] = {
    "animals": "Animals",
    "daily_life": "Everyday Life",
    "vehicles": "Explore",
}

CONFIDENCE_THRESHOLD = 0.6

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_block(x) + self.shortcut(x))


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock(32, 64, downsample=True),
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 256, downsample=True),
            ResidualBlock(256, 512, downsample=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


try:
    _checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
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
    recognized: bool
    message: str
    predicted_class: str | None
    category: str | None
    theme: str | None
    confidence: float


class ThemeItem(BaseModel):
    id: str
    label: str


class ThemesResponse(BaseModel):
    themes: list[ThemeItem]


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
    confidence_val = round(confidence.item(), 4)

    if confidence_val < CONFIDENCE_THRESHOLD:
        return ClassifyResponse(
            recognized=False,
            message="Image not recognized. Please upload again.",
            predicted_class=None,
            category=None,
            theme=None,
            confidence=confidence_val,
        )

    category = CATEGORY_MAP.get(predicted_class, "unknown")
    theme = THEME_LABELS.get(category, "Unknown")

    return ClassifyResponse(
        recognized=True,
        message="success",
        predicted_class=predicted_class,
        category=category,
        theme=theme,
        confidence=confidence_val,
    )


@router.get("/themes", response_model=ThemesResponse)
def get_themes():
    return ThemesResponse(
        themes=[ThemeItem(id=k, label=v) for k, v in THEME_LABELS.items()]
    )