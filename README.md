# Fish Species Classification using Vision Transformer and VGG16

This project implements a hybrid deep learning model that combines a Vision Transformer (ViT) and VGG16 to classify fish species from images. The model leverages the complementary strengths of both architectures to achieve high classification accuracy.

## Model Architecture

The model uses a combination of two powerful vision architectures:

1. **VGG16**: A classic convolutional neural network pretrained on ImageNet, providing robust feature extraction capabilities.
2. **Vision Transformer (ViT)**: A transformer-based architecture that processes image patches as sequences, capturing global relationships within images.

These two feature extractors are combined through:
- Global average pooling layers to reduce dimensions
- Dense layers with ReLU activation and dropout for regularization
- A final softmax layer for classification

## Dataset

The model was trained on the [Fish Species Image Dataset](https://www.kaggle.com/sripaadsrinivasan/fish-species-image-data), focusing on the top 10 most common species:
- Bodianus
- Cephalopholis
- Cirrhilabrus
- Coris
- Epinephelus
- Halichoeres
- Lethrinus
- Lutjanus
- Pseudanthias
- Thalassoma

## Training Details

- **Data Preprocessing**: Images were resized to 224×224 pixels and normalized
- **Data Augmentation**: Rotation, shifting, shearing, zooming, and horizontal flipping
- **Training Strategy**: 20 epochs with Adam optimizer (learning rate = 0.0001)
- **Batch Size**: 32
- **Loss Function**: Sparse categorical crossentropy

## Performance Metrics

The model achieved excellent performance with:
- **Test Accuracy**: 91%
- **Average AUC**: 0.99 (across all classes)

### Precision, Recall, and F1-Score per Class

| Species      | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Bodianus     | 0.95      | 0.75   | 0.84     | 28      |
| Cephalopholis| 0.85      | 0.90   | 0.88     | 31      |
| Cirrhilabrus | 1.00      | 0.85   | 0.92     | 13      |
| Coris        | 1.00      | 0.86   | 0.92     | 14      |
| Epinephelus  | 0.91      | 0.92   | 0.91     | 63      |
| Halichoeres  | 0.82      | 0.98   | 0.89     | 41      |
| Lethrinus    | 0.96      | 1.00   | 0.98     | 26      |
| Lutjanus     | 0.92      | 0.97   | 0.95     | 63      |
| Pseudanthias | 0.95      | 0.95   | 0.95     | 38      |
| Thalassoma   | 0.96      | 0.80   | 0.87     | 30      |
| **Macro Avg**| 0.93      | 0.90   | 0.91     | 347     |
| **Weighted Avg**| 0.92   | 0.91   | 0.91     | 347     |

## Key Findings

1. **Class Performance**: The model performed exceptionally well on most classes, with Lethrinus achieving a perfect recall score of 1.00 and Cirrhilabrus achieving a precision of 1.00.

2. **Confusion Patterns**: The confusion matrix reveals that:
   - Bodianus is occasionally confused with Epinephelus
   - Thalassoma shows some confusion with Halichoeres
   - Most other species have minimal confusion with accuracy above 85%

3. **Model Convergence**: Training and validation curves show good convergence with minimal overfitting. The model achieves over 90% accuracy on the training set and approximately 89% on the validation set.

4. **ROC Curves**: All classes demonstrate excellent ROC curves with AUC values between 0.98-1.00, indicating the model's strong discriminative ability across all species.

5. **Precision-Recall Performance**: The precision-recall curves show that the model maintains high precision even as recall increases, with average precision scores above 0.89 for all classes.

## Dependencies

- TensorFlow
- Transformers (Hugging Face)
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Usage

The model can be used for fish species identification in marine biology research, fishing industry applications, and ecological monitoring systems.

## Future Work

- Expand the model to include more fish species
- Test performance on underwater images with varying lighting conditions
- Implement real-time classification for video streams
- Explore knowledge distillation to create more lightweight models for mobile applications
