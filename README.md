# MRI Image Segmentation for Heart Disease Prediction

A deep learning-based project for segmenting cardiac MRI images to support heart disease diagnosis and prediction. This repository implements advanced neural network architectures for automated medical image analysis.

## Overview

Magnetic Resonance Imaging (MRI) is a crucial diagnostic tool for cardiovascular diseases. This project leverages deep learning techniques to automatically segment cardiac structures from MRI scans, enabling more efficient and accurate heart disease assessment. Automated segmentation reduces the time-intensive manual annotation process and provides consistent, reproducible results for clinical decision-making.

## Features

- **Automated MRI Segmentation**: Deep learning models for automatic cardiac structure identification
- **Multiple Architecture Support**: Implementation of state-of-the-art segmentation networks (U-Net, ResNet-based architectures)
- **Heart Disease Prediction**: Segmentation-based feature extraction for cardiovascular risk assessment
- **Image Preprocessing Pipeline**: Standardization, normalization, and augmentation techniques
- **Visualization Tools**: Display segmentation results and prediction metrics
- **Performance Metrics**: Dice coefficient, IoU, and other segmentation quality measures

## Prerequisites

```bash
Python 3.7+
PyTorch or TensorFlow
NumPy
OpenCV
Matplotlib
scikit-learn
nibabel (for medical imaging formats)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Bharathsahil-07/MRI-Image-Segmentation.git
cd MRI-Image-Segmentation
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset (if applicable):
```bash
# Add instructions for dataset acquisition
# Example: kaggle datasets download -d dataset-name
```

## Project Structure

```
MRI-Image-Segmentation/
├── data/                    # Dataset directory
│   ├── train/              # Training images and masks
│   ├── val/                # Validation data
│   └── test/               # Test data
├── models/                 # Model architectures
│   ├── unet.py            # U-Net implementation
│   ├── resnet.py          # ResNet-based segmentation
│   └── custom_models.py   # Custom architectures
├── utils/                  # Utility functions
│   ├── preprocessing.py   # Image preprocessing
│   ├── augmentation.py    # Data augmentation
│   └── metrics.py         # Evaluation metrics
├── train.py               # Training script
├── test.py                # Evaluation script
├── predict.py             # Inference on new images
├── config.yaml            # Configuration parameters
└── requirements.txt       # Project dependencies
```

## Usage

### Training

Train the segmentation model on your MRI dataset:

```bash
python train.py --config config.yaml --epochs 100 --batch_size 16
```

Key training parameters:
- `--model`: Choose architecture (unet, resnet, etc.)
- `--learning_rate`: Initial learning rate
- `--loss_function`: Segmentation loss (dice_loss, bce, combined)
- `--augmentation`: Enable/disable data augmentation

### Testing

Evaluate model performance on test data:

```bash
python test.py --model_path checkpoints/best_model.pth --test_dir data/test
```

### Prediction

Segment new MRI images:

```bash
python predict.py --image_path path/to/mri_scan.nii --output_dir results/
```

## Model Architectures

### U-Net
Classic encoder-decoder architecture with skip connections, ideal for medical image segmentation with limited training data.

### ResNet-based Segmentation
Utilizes residual connections for deeper networks, improving gradient flow and segmentation accuracy on complex cardiac structures.

### Custom Architectures
Modified networks incorporating attention mechanisms, batch normalization, and dropout for enhanced performance.

## Dataset

This project works with cardiac MRI datasets containing:
- **Input**: Multi-modal MRI scans (T1-weighted, T2-weighted, FLAIR sequences)
- **Labels**: Ground truth segmentation masks for cardiac structures (left ventricle, right ventricle, myocardium)
- **Format**: NIfTI (.nii), DICOM, or standard image formats

### Recommended Datasets
- ACDC (Automated Cardiac Diagnosis Challenge)
- Sunnybrook Cardiac Data
- UK Biobank Cardiac Imaging
- MICCAI Cardiac Challenge datasets

## Preprocessing Pipeline

1. **Image Loading**: Load MRI scans in various formats
2. **Normalization**: Standardize intensity values across scans
3. **Resampling**: Ensure consistent voxel spacing
4. **Cropping/Padding**: Standardize image dimensions
5. **Augmentation**: Rotation, flipping, elastic deformation (training only)

## Evaluation Metrics

- **Dice Similarity Coefficient (DSC)**: Measures overlap between predicted and ground truth
- **Intersection over Union (IoU)**: Jaccard index for segmentation quality
- **Hausdorff Distance**: Maximum distance between predicted and true boundaries
- **Precision/Recall**: Classification metrics for pixel-wise accuracy

## Heart Disease Prediction

After segmentation, extracted features are used for disease prediction:
- Ventricular volume measurements
- Ejection fraction calculation
- Wall thickness analysis
- Regional motion abnormality detection

Classification models predict conditions such as:
- Cardiomyopathy
- Myocardial infarction
- Heart failure
- Coronary artery disease

## Results

### Segmentation Performance
| Model | Dice Score | IoU | Training Time |
|-------|-----------|-----|---------------|
| U-Net | 0.XX | 0.XX | XX hours |
| ResNet | 0.XX | 0.XX | XX hours |

### Disease Prediction Accuracy
| Condition | Accuracy | Sensitivity | Specificity |
|-----------|----------|-------------|-------------|
| Condition 1 | XX% | XX% | XX% |
| Condition 2 | XX% | XX% | XX% |

## Visualization

The project includes visualization tools for:
- Original MRI slices with segmentation overlays
- 3D volume rendering of cardiac structures
- Training loss and accuracy curves
- Confusion matrices for disease classification

## Future Work

- [ ] Integration of multi-modal imaging (MRI + CT fusion)
- [ ] Real-time segmentation for clinical deployment
- [ ] Explainable AI features for clinical interpretability
- [ ] Transfer learning from larger medical imaging datasets
- [ ] Web-based interface for radiologist interaction

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Medical imaging datasets provided by [Dataset Source]
- Pre-trained models and architectures inspired by medical imaging research community
- Clinical consultation and validation from domain experts

## Contact

For questions, issues, or collaboration opportunities:
- GitHub: [@Bharathsahil-07](https://github.com/Bharathsahil-07)
- Email: bharathsahil635@gmail.com

## References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
2. Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation" (2018)
3. Chen et al., "Deep Learning for Cardiac Image Segmentation: A Review" (2020)

---

**Disclaimer**: This software is for research purposes only and should not be used for clinical diagnosis without proper validation and regulatory approval.
