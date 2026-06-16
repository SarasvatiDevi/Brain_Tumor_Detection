import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical

# ======== SETUP ========
IMAGE_SIZE = 224
BATCH_SIZE = 12
EPOCHS = 10
train_dir = 'C:/Users/User/PycharmProjects/Brain_Tumor_Detection/MRI_Images/Training'
test_dir = 'C:/Users/User/PycharmProjects/Brain_Tumor_Detection/MRI_Images/Testing'

classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(classes)

# ======== LOAD DATA ========
def load_dataset(dir_path):
    paths, labels = [], []
    for label in classes:
        full_path = os.path.join(dir_path, label)
        for image in os.listdir(full_path):
            paths.append(os.path.join(full_path, image))
            labels.append(label)
    return shuffle(paths, labels)

train_paths, train_labels = load_dataset(train_dir)
test_paths, test_labels = load_dataset(test_dir)

# ======== IMAGE AUGMENTATION ========
def augment_image(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return np.array(image) / 255.0

def open_images(paths, augment=True):
    images = []
    for path in paths:
        img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        if augment:
            img = augment_image(img)
        else:
            img = np.array(img) / 255.0
        images.append(img)
    return np.array(images)

# ======== LABEL ENCODING ========
def encode_labels(labels):
    label_to_index = {label: i for i, label in enumerate(classes)}
    encoded = [label_to_index[label] for label in labels]
    return to_categorical(encoded, num_classes=num_classes)

# ======== DATA GENERATOR ========
def datagen(paths, labels, batch_size, epochs=1):
    for _ in range(epochs):
        paths, labels = shuffle(paths, labels)  # shuffle each epoch
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_labels_raw = labels[i:i + batch_size]
            batch_images = open_images(batch_paths)
            batch_labels = encode_labels(batch_labels_raw)
            yield batch_images, batch_labels

# ======== MODEL DEFINITION ========
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-4:]:
    layer.trainable = True

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ======== TRAINING ========
steps = len(train_paths) // BATCH_SIZE
history = model.fit(
    datagen(train_paths, train_labels, BATCH_SIZE, EPOCHS),
    steps_per_epoch=steps,
    epochs=EPOCHS
)

# ======== TRAINING PLOT ========
plt.figure(figsize=(8, 4))
plt.plot(history.history['categorical_accuracy'], 'g-', label='Accuracy')
plt.plot(history.history['loss'], 'r-', label='Loss')
plt.title("Training Accuracy & Loss")
plt.xlabel("Epochs")
plt.grid(True)
plt.legend()
plt.show()

# ======== EVALUATION ========
test_images = open_images(test_paths, augment=False)
test_labels_encoded = encode_labels(test_labels)
predictions = model.predict(test_images)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_encoded, axis=1)

print("Classification Report:\n", classification_report(true_classes, predicted_classes, target_names=classes))

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ======== ROC CURVE ========
test_labels_bin = label_binarize(true_classes, classes=np.arange(num_classes))
fpr, tpr, roc_auc = {}, {}, {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ======== SAVE MODEL ========
model_path = 'C:/Users/User/PycharmProjects/Brain_Tumor_Detection/image_classifier_model.h5'
try:
    print("Saving model...")
    model.save(model_path)
    if os.path.exists(model_path):
        print(f"✅ Model saved successfully at: {model_path}")
    else:
        print("⚠️ Model.save() completed but file not found. Please check your path or permissions.")
except Exception as e:
    print("❌ Failed to save model:", e)