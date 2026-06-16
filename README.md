# Brain Tumor MRI Classification

A deep learning web application that classifies brain MRI scans into 4 tumor categories (Glioma, Meningioma, Pituitary, No Tumor) using a fine-tuned VGG16 model trained on 7200  MRI images — achieving 97% accuracy and deployed with Flask for real-time clinical inference.
---

<img width="763" height="471" alt="image" src="https://github.com/user-attachments/assets/2d6c97f2-cfac-4ee4-a1f5-dbfa0378d4fb" />

<img width="790" height="474" alt="image" src="https://github.com/user-attachments/assets/3963b58d-aaa6-4299-8ecb-8915b612290d" />



---

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **97%** |
| Macro Avg Precision | 0.97 |
| Macro Avg Recall | 0.97 |
| Macro Avg F1-Score | 0.97 |
| Test Set Size | 1,311 images |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |

|-------|-----------|--------|----------|---------|

| Glioma | 0.97 | 0.96 | 0.96 | 300 |

| Meningioma | 0.92 | 0.96 | 0.94 | 306 |

| No Tumor | 1.00 | 0.99 | 1.00 | 405 |

| Pituitary | 0.99 | 0.97 | 0.98 | 300 |

---

## How It Works

1. User uploads a brain MRI image via the Flask web interface

2. Image is resized to 224×224 and normalized

3. Fine-tuned VGG16 model predicts the tumor class

4. Result is returned with confidence percentage in real time

---

## Model Architecture

- **Base Model:** VGG16 (pretrained on ImageNet)

- **Fine-tuning:** Last 4 layers unfrozen for domain adaptation

- **Custom Head:** Flatten → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(4, Softmax)

- **Optimizer:** Adam (lr=0.0001)

- **Loss:** Categorical Crossentropy

- **Input Size:** 224 × 224 × 3

- **Epochs:** 10 | **Batch Size:** 12

---

## Dataset

- **Source:** Brain Tumor MRI Dataset (Training + Testing split)

- **Classes:** Glioma, Meningioma, No Tumor, Pituitary

- **Preprocessing:** Brightness & contrast augmentation, pixel normalization (÷255)

- **Train/Test Split:** Separate training and testing directories

---

## Project Structure

```
brain-tumor-detection/

│

├── app.py                    # Flask web app for real-time inference

├── brain_tumor_detection.py  # Model training, evaluation, ROC curves

├── models/

│   └── image_classifier_model.h5   # Trained VGG16 model

├── templates/

│   └── index.html            # Web interface

├── uploads/                  # Temporary uploaded MRI images

├── requirements.txt

└── README.md
```

---

## Installation & Setup

```bash

git clone https://github.com/SarasvatiDevi/brain_tumor_detection

cd brain_tumor_detection

pip install -r requirements.txt

python app.py

```

Then open `http://localhost:5000` in your browser and upload an MRI image.

> ⚠️ The model is auto-downloaded from Google Drive on first run via `gdown`.

---

## Requirements

```
tensorflow

flask

numpy

Pillow

matplotlib

seaborn

scikit-learn

gdown

```

---

## Evaluation Visualizations

The training script generates:

- ✅ Training Accuracy & Loss curve

- ✅ Confusion Matrix (seaborn heatmap)

- ✅ Per-class ROC Curves with AUC scores


---

## Limitations

- Trained on a single benchmark dataset — performance on hospital MRI data may vary

- Web app currently runs locally only (no cloud deployment)

- Model size (~500MB) requires Google Drive hosting for distribution


## Future Work


- Deploy on Hugging Face Spaces or Render for public access

- Extend to external hospital MRI datasets for generalization testing

- Add Grad-CAM visualization to highlight tumor regions in predictions

- Integrate additional CNN architectures for ensemble inference

---

## Author

**Sarasvati Devi**
BS Artificial Intelligence — Aror University, Sukkur, Pakistan

[LinkedIn](https://linkedin.com/in/sarasvati-devi-07251b293) · [GitHub](https://github.com/SarasvatiDevi)
