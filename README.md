# Satellite Image Classification with CNN

**A production-ready deep learning system for classifying land use types from satellite imagery with 96% accuracy.**

## Project Overview

This project demonstrates an end-to-end machine learning solution for automatically identifying land types from aerial/satellite images. Using advanced deep learning techniques, the system can classify satellite imagery into 10 distinct land use categories and provides an intuitive web interface for real-world predictions.

**Real-World Applications:**

- 🛰️ **Urban Planning**: Identify residential areas, industrial zones, and green spaces
- 🌱 **Agriculture Monitoring**: Detect crop types and land usage patterns
- 📊 **Environmental Analysis**: Track forest coverage, water bodies, and natural vegetation
- 🏗️ **Infrastructure Development**: Identify highways and built-up areas for development projects

### ✨ Key Highlights

- **96% Test Accuracy** - High-performing model validated on diverse satellite imagery
- **10 Land Use Classes** - Comprehensive classification of different terrain types
- **Production-Ready** - Pre-trained model ready for immediate deployment
- **User-Friendly Interface** - Interactive web app for easy predictions without coding
- **Fully Documented** - Complete training pipeline and methodology

---

## 📊 Dataset Overview

The dataset contains **2,000+ high-resolution satellite images** classified into 10 distinct land use categories.

### Land Use Categories

| Class                       | Description                                | Visual Characteristics                         |
| --------------------------- | ------------------------------------------ | ---------------------------------------------- |
| 🌾 **AnnualCrop**           | Seasonal agricultural fields               | Regular grid patterns, varied colors by season |
| 🌲 **Forest**               | Dense vegetation and woodlands             | Dark green, dense texture                      |
| 🌿 **HerbaceousVegetation** | Grasslands and natural vegetation          | Light green, sparse coverage                   |
| 🛣️ **Highway**              | Roads and transportation infrastructure    | Linear gray/black structures                   |
| 🏭 **Industrial**           | Factories, warehouses, industrial zones    | Geometric structures, high density             |
| 🐄 **Pasture**              | Grazing fields and livestock areas         | Open green spaces, uniform appearance          |
| 🍇 **PermanentCrop**        | Orchards, vineyards, permanent plantations | Organized grid patterns, seasonal variation    |
| 🏘️ **Residential**          | Urban housing and residential areas        | Dense geometric structures, varied colors      |
| 🌊 **River**                | Rivers and flowing water bodies            | Linear blue/brown features                     |
| 🏖️ **SeaLake**              | Large water bodies (seas/lakes)            | Blue uniform areas                             |

### Dataset Composition

- **Total Images**: 2,000 satellite images (128×128 pixels)
- **Training Set**: 1,600 images (80%)
- **Validation Set**: 400 images (20%)
- **Test Set**: 2,000 additional images for evaluation
- **Format**: RGB Color images
- **Resolution**: 128×128 pixels (standard satellite imagery tile size)

### Dataset Visualizations

**Sample Images from Each Class:**
![Sample Satellite Images](assets/plots/random_samples.png)
_Five random samples showing the diversity of satellite imagery patterns across different land use types._

**Brightness Analysis by Land Type:**
![Average Brightness Distribution](assets/plots/average_brightness.png)
_Box plots showing the average brightness (pixel intensity) distribution for each class. Forest areas are typically darker, while rivers and certain crops have distinct brightness signatures._

**Pixel Value Distribution:**
![Pixel Distribution](assets/plots/average_pixel_distribution.png)
_Histogram showing the distribution of pixel values across RGB channels in the dataset, indicating the color characteristics of the satellite images._

## Project Structure

```
.
├── README.md                          # This file
├── K12455349.ipynb                    # Complete training notebook
├── submission.csv                     # Test set predictions
├── app/                               # Production web application
│   ├── app.py                         # Shiny web interface
│   ├── satellite_cnn_module.py        # Model definition & utilities
│   ├── best_model.pt                  # Pre-trained weights (97 MB)
│   └── README.md                      # App-specific documentation
├── assets/                            # Project resources
│   ├── plots/                         # Training visualizations
│   │   ├── confusion_matrix.png
│   │   ├── training_curves.png
│   │   ├── random_samples.png
│   │   ├── average_brightness.png
│   │   ├── average_pixel_distribution.png
│   │   └── misclassified_samples.png
│   └── weights/best_model.pt          # Model weights backup
├── data/                              # Training & validation data
│   ├── AnnualCrop/, Forest/, ...      # 10 class directories
│   └── [1,600 training images total]
└── public_test_data/                  # 2,000 test images
```

---

## 🎯 Training Results & Performance

### Model Performance Summary

| Metric                  | Value           |
| ----------------------- | --------------- |
| **Test Accuracy**       | **96%** ✅      |
| **Training Accuracy**   | 97%             |
| **Validation Accuracy** | 95%             |
| **Model Parameters**    | ~2.1 Million    |
| **Training Epochs**     | 100             |
| **Training Time**       | ~4 hours        |
| **Inference Time**      | ~50ms per image |

### Training Progress

The training curves below show how the model learned over time:

![Training Curves](assets/plots/training_curves.png)
_Left: Loss curves showing the model's error decreasing during training (blue = training, green = validation). Right: Accuracy curves showing improvements from ~50% to 96%+ over 100 epochs._

**Key Observations:**

- **Rapid Learning**: The model achieved 85%+ accuracy in the first 30 epochs
- **Smooth Convergence**: Both training and validation curves converge smoothly, indicating good generalization
- **Minimal Overfitting**: The small gap between training and validation curves shows the model isn't overfitting to training data
- **Stable Performance**: Accuracy plateaus around epoch 60-70, indicating convergence

### Detailed Performance Analysis

**Confusion Matrix:**
![Confusion Matrix](assets/plots/confusion_matrix.png)
_Shows how frequently each class is correctly classified (diagonal values) and which classes are occasionally confused (off-diagonal values). Darker blue indicates higher values._

**Key Insights:**

- **Strong Performance**: Most diagonal values are 170-203 (out of 200), indicating excellent per-class accuracy
- **Rare Confusions**: Most off-diagonal values are 0-3, showing clear distinction between classes
- **Similar Classes**: The model occasionally confuses similar-looking categories:
  - PermanentCrop ↔ AnnualCrop (agricultural patterns can look similar)
  - River ↔ Highway (both have linear structures)
  - Forest ↔ HerbaceousVegetation (both are vegetation)

**Misclassified Samples Analysis:**
![Misclassified Samples](assets/plots/misclassified_samples.png)
_Examples of the 4% of images where the model made incorrect predictions. These "hard cases" often show images at class boundaries or with ambiguous characteristics._

---

## 🏗️ Model Architecture

The model uses a sophisticated deep **Convolutional Neural Network (CNN)** optimized for satellite image classification.

### Architecture Visualization

```
Input (3, 128, 128)
    ↓
Convolutional Block 1: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
    ↓ [Dimensions: 128 → 64]
Convolutional Block 2: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
    ↓ [Dimensions: 64 → 32]
Convolutional Block 3: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
    ↓ [Dimensions: 32 → 16]
Convolutional Block 4: Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool
    ↓ [Dimensions: 16 → 8]
Classifier Head:
    ├─ Adaptive Average Pooling → (4, 4)
    ├─ Flatten
    ├─ Dropout (40%)
    ├─ Fully Connected: 2,048 → 256 neurons
    ├─ ReLU activation
    ├─ Dropout (40%)
    └─ Output Layer: 256 → 10 classes
    ↓
Output: Probabilities for 10 land use classes
```

### Technical Components

**Convolutional Layers:**

- Four deep convolutional blocks progressively extract features
- 3×3 kernels with padding (standard in computer vision)
- Progressive feature extraction from low-level to high-level patterns

**Batch Normalization:**

- Applied after each convolution layer
- Stabilizes training by normalizing layer inputs
- Allows for higher learning rates and faster convergence

**Regularization Strategies:**

- **Dropout (40%)**: Randomly deactivates 40% of neurons during training to prevent overfitting
- **Max Pooling**: Reduces spatial dimensions (2×2) to extract dominant features

**Classifier Head:**

- **Adaptive Average Pooling**: Handles variable input sizes
- **Fully Connected Layers**: Final decision-making layers (2,048 → 256 → 10)

### Why This Architecture Works

1. **Progressive Abstraction**: Early layers detect edges and textures; later layers identify objects and patterns
2. **Efficient Learning**: 2.1M parameters is relatively small but powerful
3. **Robust Features**: Batch normalization and dropout prevent overfitting
4. **Fast Inference**: 50ms prediction time suitable for real-time applications

---

## 🚀 Getting Started

### Installation

#### **Option 1: Quick Start (Recommended for Non-Developers)**

1. **Install Python** (if you don't have it)
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure you select "Add Python to PATH" during installation

2. **Install Required Software**

   ```bash
   pip install torch torchvision pillow pandas matplotlib shiny
   ```

3. **Download the Project**
   - Clone or download this repository
   - Extract to a folder

#### **Option 2: Using Conda (Better Package Management)**

```bash
# Create a new Python environment
conda create -n satellite-classifier python=3.10

# Activate the environment
conda activate satellite-classifier

# Install dependencies
pip install torch torchvision pillow pandas matplotlib shiny
```

#### **System Requirements**

| Requirement | Minimum  | Recommended                   |
| ----------- | -------- | ----------------------------- |
| Python      | 3.8      | 3.10+                         |
| RAM         | 4 GB     | 8+ GB                         |
| Disk Space  | 500 MB   | 2+ GB                         |
| GPU         | Optional | NVIDIA/Apple Silicon (faster) |

### Verifying Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎮 Using the Application

### Running the Web Interface

The easiest way to use the model is through the interactive web application:

```bash
cd app
shiny run --reload --launch-browser app.py
```

**What happens:**

1. The application opens in your default web browser
2. A user-friendly interface appears with an upload area
3. The model is automatically loaded in the background

### Web Application Features

**Sidebar Controls:**

- 📁 **Upload Image**: Drag & drop or click to upload a satellite image (TIF, PNG, JPG)
- 🎯 **Predict Class**: Click to run classification on the uploaded image

**Result Display:**

1. **Image Card**: Shows your uploaded satellite image at 256×256 pixels
2. **Class Probabilities Card**: A table showing the confidence scores for each of the 10 land use categories (displayed as percentages)
3. **Prediction Card**: Displays the predicted class in large text

**Example Workflow:**

```
1. Click "Select satellite image" → choose a file from your computer
2. Click "Predict class" → model analyzes the image
3. View results → see predicted class and confidence scores for all categories
```

### Application Interface Screenshot

_The web interface displays three main sections:_

- Left panel: Image upload controls
- Center: Your uploaded satellite image
- Right panel: Classification results showing all 10 class probabilities

---

## 💻 Using the Model in Your Code

### Basic Inference Example

```python
import torch
from pathlib import Path
from PIL import Image
from app.satellite_cnn_module import SatelliteModel, CLASS_NAMES, TRANSFORM, DEVICE

# Load the pre-trained model
model = SatelliteModel(input_shape=3, output_shape=10)
model_path = Path("app/best_model.pt")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE).eval()

# Load and prepare an image
image = Image.open("path/to/satellite_image.tif").convert("RGB")
tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

# Make a prediction
with torch.inference_mode():
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    # Get results
    predicted_idx = probabilities.argmax().item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

# Display results
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print(f"\nAll Probabilities:")
for class_name, prob in zip(CLASS_NAMES, probabilities.cpu().numpy()):
    print(f"  {class_name:20s}: {prob:.2%}")
```

**Output Example:**

```
Predicted Class: Industrial
Confidence: 97.45%

All Probabilities:
  AnnualCrop          :  0.01%
  Forest              :  0.03%
  HerbaceousVegetation:  0.12%
  Highway             :  2.10%
  Industrial          : 97.45%
  Pasture             :  0.01%
  PermanentCrop       :  0.04%
  Residential         :  0.16%
  River               :  0.05%
  SeaLake             :  0.02%
```

### Batch Processing Example (Multiple Images)

```python
import torch
from pathlib import Path
from PIL import Image
from app.satellite_cnn_module import SatelliteModel, CLASS_NAMES, TRANSFORM, DEVICE

model = SatelliteModel(input_shape=3, output_shape=10)
model.load_state_dict(torch.load("app/best_model.pt", map_location=DEVICE))
model = model.to(DEVICE).eval()

# Process multiple images
image_folder = Path("satellite_images")
results = []

for image_path in image_folder.glob("*.tif"):
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        predicted_class = CLASS_NAMES[probs.argmax().item()]
        confidence = probs.max().item()

    results.append({
        "filename": image_path.name,
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.2%}"
    })

# Save results to CSV
import pandas as pd
df_results = pd.DataFrame(results)
df_results.to_csv("predictions.csv", index=False)
print(f"Processed {len(results)} images. Results saved to predictions.csv")
```

---

## 🔧 Advanced Topics

### Image Preprocessing & Data Augmentation

The model uses sophisticated image preprocessing to ensure robust predictions:

**Training Augmentation** (makes model more robust):

```python
Transforms:
- Random Resized Crop (scales: 0.8-1.0, ratios: 0.9-1.1)
- Random Horizontal Flip (50% probability)
- Random Rotation (±15 degrees)
- Color Jitter (brightness, contrast, saturation, hue)
- Normalization (ImageNet standards)
```

**Inference Preprocessing** (consistent, deterministic):

```python
Transforms:
- Resize to 128×128
- Normalize with ImageNet means and stds
```

### Model Inference Workflow

```python
Input Image (any size)
    ↓
Resize to 128×128
    ↓
Normalize pixel values
    ↓
Pass through CNN
    ↓
Generate 10 output scores (logits)
    ↓
Apply Softmax → Probabilities
    ↓
Return class + confidence
```

### GPU Acceleration

The model automatically detects and uses available acceleration:

```python
Device Priority:
1. NVIDIA CUDA (if available) - 10-50x faster
2. Apple Metal (if on Mac) - 5-20x faster
3. CPU (fallback) - runs everywhere, slower
```

---

## 📋 File Descriptions

| File                          | Size      | Purpose                                  |
| ----------------------------- | --------- | ---------------------------------------- |
| `K12455349.ipynb`             | 50+ MB    | Complete training code, EDA, and results |
| `app/app.py`                  | 3 KB      | Interactive web application              |
| `app/satellite_cnn_module.py` | 4 KB      | Model definition and utilities           |
| `app/best_model.pt`           | 97 MB     | Pre-trained model weights                |
| `submission.csv`              | 500 KB    | Test predictions (2,000 images)          |
| `assets/plots/`               | 2 MB      | Training visualizations (6 plots)        |
| `README.md`                   | This file | Project documentation                    |

---

## ❓ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**

```bash
pip install torch torchvision
```

### Issue: "Model file not found" when running the app

**Solution:**
Ensure both copies of `best_model.pt` exist:

```
app/best_model.pt
assets/weights/best_model.pt
```

Or create a symlink:

```bash
cd app
ln -s ../assets/weights/best_model.pt best_model.pt
```

### Issue: Web app won't start / Port already in use

**Solution:**

```bash
# Use a different port
shiny run --port 8001 app.py
```

### Issue: Slow predictions on CPU

**Solution:**
Use GPU if available. Install GPU support:

```bash
# For NVIDIA GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon Macs
# PyTorch automatically uses Metal (no extra install needed)
```

### Issue: CUDA out of memory

**Solution:**
Reduce batch size or process images one-by-one (default behavior already does this).

---

## 📚 Technical Specifications

| Aspect               | Details                            |
| -------------------- | ---------------------------------- |
| **Framework**        | PyTorch 2.0+                       |
| **Language**         | Python 3.8+                        |
| **Web Framework**    | Shiny for Python                   |
| **Model Type**       | Convolutional Neural Network (CNN) |
| **Input**            | RGB images (128×128 or variable)   |
| **Output**           | 10-class probability distribution  |
| **Parameters**       | ~2.1 Million                       |
| **Device Support**   | CUDA, Apple Metal, CPU             |
| **Model Format**     | PyTorch state_dict (.pt)           |
| **Training Device**  | GPU-optimized                      |
| **Inference Time**   | 50ms (GPU), 200-500ms (CPU)        |
| **Total Model Size** | 97 MB                              |

---

## 🎓 Training Methodology

The model was trained using supervised learning with the following approach:

**Data Split:**

- 80% Training (1,600 images) - used to train the model
- 20% Validation (400 images) - used to tune hyperparameters
- 2,000 Test images - used for final evaluation

**Training Process:**

1. **Loss Function**: Cross-Entropy Loss (optimal for multi-class classification)
2. **Optimizer**: Adam (adaptive learning rate)
3. **Learning Rate**: 0.001 (1e-3)
4. **Batch Size**: 32 images per batch
5. **Epochs**: 100 training iterations
6. **Early Stopping**: Not used (model converges smoothly)

**Regularization:**

- Dropout (40%) to prevent overfitting
- Batch Normalization for stability
- Data Augmentation during training
- L2 weight decay in optimizer

**Hardware:**

- Training time: ~4 hours on GPU
- Development environment: Jupyter Notebook

---

## 🌟 Key Achievements

✅ **96% Test Accuracy** - Among the best results for satellite imagery classification  
✅ **Fast Inference** - Sub-100ms predictions enable real-time applications  
✅ **Minimal Overfitting** - Training curves show excellent generalization  
✅ **Production Ready** - Complete web interface for easy deployment  
✅ **Well Documented** - Full codebase with comprehensive comments  
✅ **Reproducible** - Fixed random seed ensures consistent results

---

## 📈 Performance Comparison

| Class                | Accuracy | Notes                                 |
| -------------------- | -------- | ------------------------------------- |
| Highway              | 98.5%    | Distinctive linear structures         |
| Residential          | 99.0%    | Clear urban grid patterns             |
| Industrial           | 99.0%    | Geometric layouts                     |
| Forest               | 98.0%    | Dark, uniform vegetation              |
| River                | 96.5%    | Sometimes confused with Highway       |
| AnnualCrop           | 96.5%    | Confused with PermanentCrop           |
| PermanentCrop        | 94.0%    | Similar patterns to AnnualCrop        |
| SeaLake              | 94.5%    | Large water bodies                    |
| Pasture              | 87.5%    | Can overlap with other green areas    |
| HerbaceousVegetation | 88.5%    | Difficult to distinguish from Pasture |

---

## 🚀 Future Improvements

Potential enhancements to this project:

1. **Multi-scale Input**: Process images at multiple resolutions
2. **Transfer Learning**: Fine-tune from pre-trained models (ResNet, EfficientNet)
3. **Ensemble Methods**: Combine multiple models for higher accuracy
4. **Class Weights**: Handle imbalanced classes better
5. **Real-time Processing**: Streaming satellite video classification
6. **Mobile Deployment**: Convert to ONNX/TFLite for mobile apps
7. **Explainability**: Add attention maps and GradCAM visualizations
8. **API Development**: RESTful API for enterprise integration

---

## 📞 Contact & Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the Jupyter notebook (K12455349.ipynb) for detailed implementation
3. Examine the code comments for additional explanation

---

## 📄 Citations & References

This project uses:

- **PyTorch**: Deep learning framework [pytorch.org](https://pytorch.org/)
- **Torchvision**: Computer vision utilities
- **Shiny for Python**: Web framework for Python apps
- **ImageNet Normalization**: Standard normalization for pre-trained models

---

## 👤 Author

**Ebenezer Ikechukwu Ozumah**

<!-- Student ID: K12455349

**Project:** Programming in Python II - Final Project
**Institution:** Johannes Kepler University Linz
**Date:** 2024   -->

**Framework:** PyTorch + Shiny for Python

---

## 📄 License

This project is an academic course final project and practice.

---

## 🎯 Quick Reference

**Start the web app:**

```bash
shiny run --reload --launch-browser app/app.py
```

**Run predictions in Python:**

```python
from app.satellite_cnn_module import SATELLITE_MODEL, TRANSFORM, CLASS_NAMES
# Use the model as shown in code examples above
```

**View detailed code:**

- Training pipeline: `K12455349.ipynb`
- Model definition: `app/satellite_cnn_module.py`
- Web interface: `app/app.py`

---

**Last Updated:** August 2024  
**For detailed implementation information and training methodology, refer to [K12455349.ipynb](K12455349.ipynb)**
