# Crop Disease Prediction Model

An automated crop disease prediction system using deep learning with transfer learning and MobileNet architecture. This project enables farmers and agronomists to rapidly identify crop diseases from leaf images with high accuracy.

## Features

- **38 Disease Classifications**: Detects and classifies 38 crop diseases and healthy leaf conditions
- **High Accuracy**: Achieves ~96% validation accuracy with robust per-class performance
- **Efficient Architecture**: Uses MobileNet with transfer learning for fast inference
- **Large Dataset**: Trained on over 70,000 augmented crop leaf images
- **Multiple Formats**: Model available in `.keras`, `.h5`, and `.tflite` formats
- **Production Ready**: Easy integration into web, cloud, and mobile environments
- **Well-Documented**: Includes preprocessing scripts and deployment guidelines

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Deployment](#deployment)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements

- Python 3.10 or higher
- TensorFlow 2.x
- Keras
- Pillow (PIL)
- NumPy
- Matplotlib
- Scikit-learn (for evaluation metrics)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd crop-disease-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

### Structure

The dataset is organized into three main directories:

```
data/
├── train/
│   ├── disease_1/
│   ├── disease_2/
│   └── ... (38 classes)
├── validation/
│   ├── disease_1/
│   ├── disease_2/
│   └── ... (38 classes)
└── test/
    ├── disease_1/
    ├── disease_2/
    └── ... (38 classes)
```

### Data Characteristics

- **Total Images**: 70,000+
- **Image Resolution**: High-resolution crop leaf images
- **Classes**: 38 (37 diseases + 1 healthy class)
- **Augmentation**: Real-time data augmentation during training (shear, zoom, shifts)
- **Format**: JPG/PNG

### Data Augmentation

Applied transformations include:
- Rescaling (1/255.0)
- Shear range: 0.2
- Zoom range: 0.2
- Width shift: 0.2
- Height shift: 0.2
- Fill mode: nearest

## Model Architecture

### Base Architecture

- **Backbone**: MobileNet (pretrained on ImageNet, top layers removed)
- **Input Size**: 224x224x3 (RGB)
- **Output**: 38 classes (softmax activation)

### Custom Layers

```
MobileNet (pretrained)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.5)
    ↓
Dense (128, activation='relu')
    ↓
Dense (38, activation='softmax')
```

### Model Parameters

- **Total Parameters**: ~4.2M (efficient for mobile deployment)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score

## Training

### Training Configuration

```python
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest"
)

# Load training data
train_data = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Validation data (no augmentation)
val_datagen = ImageDataGenerator(rescale=1/255.0)
val_data = val_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)
```

### Training Parameters

- **Epochs**: 50 (with early stopping)
- **Batch Size**: 32
- **Validation Split**: 20%
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Training Time**: ~2-4 hours (GPU recommended)

### Training Results

- **Final Training Accuracy**: 98.5%
- **Validation Accuracy**: 96.0%
- **No Overfitting**: Stable convergence with early stopping at epoch 42

## Inference

### Basic Inference

```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('models/best_model.h5')

# Load and preprocess image
img = Image.open('path/to/leaf_image.jpg')
img = img.resize((224, 224))
x = np.array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Make prediction
prediction = model.predict(x)
predicted_class = np.argmax(prediction[0])
confidence = np.max(prediction[0])
```

### Batch Inference

```python
# Predict on multiple images
images = []
for img_path in image_paths:
    img = Image.open(img_path).resize((224, 224))
    images.append(np.array(img) / 255.0)

images = np.array(images)
predictions = model.predict(images)
```

### Using TFLite for Mobile/Edge

```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='models/model.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])
```

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | 96.0% |
| Macro-Avg Precision | 0.96 |
| Macro-Avg Recall | 0.96 |
| Macro-Avg F1-Score | 0.96 |
| Per-Class Accuracy | >0.94 for 35/38 classes |

### Loss & Accuracy Curves

- Training curves show stable convergence without overfitting
- Early stopping triggered at epoch 42
- Validation accuracy plateaus around 96% with no degradation

### Per-Class Performance

Detailed classification reports available in `/results/classification_report.csv`:
- Precision, Recall, F1-Score for each of 38 classes
- Support (number of test samples per class)
- Weighted averages accounting for class imbalance

## Deployment

### Export Formats

The model is exported in multiple formats for different deployment scenarios:

1. **Keras Format** (`.keras`): Full model with architecture and weights
   ```bash
   model.save('models/best_model.keras')
   ```

2. **HDF5 Format** (`.h5`): Compatible with legacy TensorFlow versions
   ```bash
   model.save('models/best_model.h5')
   ```

3. **TFLite Format** (`.tflite`): Optimized for mobile and edge devices
   ```bash
   converter = tf.lite.TFLiteConverter.from_saved_model('models/saved_model')
   tflite_model = converter.convert()
   ```

### Class Mapping

Use `class_indices.json` for human-readable predictions:

```json
{
  "0": "Apple___Apple_scab",
  "1": "Apple___Black_rot",
  "2": "Apple___Cedar_apple_rust",
  ...
  "37": "healthy"
}
```

### Web Deployment (Flask Example)

```python
from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import json

app = Flask(__name__)
model = keras.models.load_model('models/best_model.h5')

with open('class_indices.json') as f:
    class_labels = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(file).resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    prediction = model.predict(x)
    predicted_class_idx = str(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0]))
    disease_name = class_labels[predicted_class_idx]
    
    return jsonify({
        'disease': disease_name,
        'confidence': confidence,
        'class_index': predicted_class_idx
    })

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Mobile Deployment (TensorFlow Lite)

For Android/iOS applications, use the `.tflite` model with TensorFlow Lite runtime libraries.

## Usage Examples

### Example 1: Single Image Prediction

```python
from tensorflow import keras
from PIL import Image
import numpy as np
import json

# Load model and class labels
model = keras.models.load_model('models/best_model.h5')
with open('class_indices.json') as f:
    class_labels = json.load(f)

# Load image
img = Image.open('samples/apple_leaf.jpg')
img = img.resize((224, 224))
x = np.array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Predict
prediction = model.predict(x)
predicted_idx = str(np.argmax(prediction[0]))
confidence = np.max(prediction[0])

print(f"Disease: {class_labels[predicted_idx]}")
print(f"Confidence: {confidence:.2%}")
```

### Example 2: Directory Batch Processing

```python
import os
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd

model = keras.models.load_model('models/best_model.h5')
with open('class_indices.json') as f:
    class_labels = json.load(f)

results = []
image_dir = 'samples/'

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path).resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    prediction = model.predict(x, verbose=0)
    predicted_idx = str(np.argmax(prediction[0]))
    
    results.append({
        'image': img_name,
        'disease': class_labels[predicted_idx],
        'confidence': np.max(prediction[0])
    })

df = pd.DataFrame(results)
df.to_csv('predictions.csv', index=False)
```

## Project Structure

```
crop-disease-prediction/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── models/
│   ├── best_model.h5
│   ├── best_model.keras
│   ├── model.tflite
│   └── class_indices.json
├── results/
│   ├── training_history.png
│   ├── loss_accuracy_curves.png
│   └── classification_report.csv
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── inference.py
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request


## Acknowledgments

- TensorFlow and Keras communities
- Dataset contributors and agricultural researchers
- Open-source community for tools and libraries
- AgriBazaar Project

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [vijaykumarputta08@gmail.com]
- Documentation: [link to docs]

---

