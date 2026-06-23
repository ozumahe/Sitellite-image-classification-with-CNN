# Initializes the satellite model and all the utils needed for the app.
import torch # type: ignore
import torch.nn as nn # type: ignore
from torchvision import transforms # type: ignore
from pathlib import Path

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Satellite model
class SatelliteModel(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units=128):
        """Initializes the Satellite multi-class classification model

        Args:
            input_shape (tuple): The shape of the input images (C, H, W).
            output_shape (int): The number of output classes.
            hidden_units (int, optional): The number of hidden units in the fully connected layer.-- defaults: 128.
        """
        super().__init__()
        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.convolutional_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.convolutional_block_4 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_units * 4 * 4, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the model

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: The output tensor of shape (B, output_shape)
        """
        x = self.convolutional_block_1(x)
        # print(f"Shape after convolutional_block_1: {x.shape}")

        x = self.convolutional_block_2(x)
        # print(f"Shape after convolutional_block_2: {x.shape}")

        x = self.convolutional_block_3(x)
        # print(f"Shape after convolutional_block_3: {x.shape}")

        x = self.convolutional_block_4(x)
        # print(f"Shape after convolutional_block_4: {x.shape}")

        x = self.classifier(x)
        return x
    

# Device agnostic code
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

# Module input transformer
TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SATELLITE_MODEL = SatelliteModel(input_shape=3, hidden_units=128, output_shape=len(CLASS_NAMES)).to(DEVICE)

print(SATELLITE_MODEL, "\n")
print(f"{SATELLITE_MODEL.__class__.__name__} on device --> {next(SATELLITE_MODEL.parameters()).device}")

# Load saved model parameters
MODEL_SAVED_PATH = Path(__file__).parents[1] / "assets" / "weights" / 'best_model.pt'

if not MODEL_SAVED_PATH.exists():
    MODEL_SAVED_PATH = Path(__file__).parent / 'best_model.pt'
    
state_dict = torch.load(MODEL_SAVED_PATH, map_location=DEVICE)

# Set the loaded state_dict to the model and set the model to evaluation mode.
_ = SATELLITE_MODEL.load_state_dict(state_dict)
_ = SATELLITE_MODEL.eval()
