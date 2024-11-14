import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Define the directories for benign and malignant images
image_dirs = [
    r"C:\Users\win10\Downloads\breastcancer\BreaKHis_v1\histology_slides\breast\breastCancer\benign"
    ,
    r"C:\Users\win10\Downloads\breastcancer\BreaKHis_v1\histology_slides\breast\breastCancer\malignant"
]

# Function to load and resize images (handling potential errors)
def load_and_process_image(file_path, label):
    try:
        with Image.open(file_path) as image:
            # Resize image to 128x128 for faster processing
            image = image.resize((128, 128))  # Smaller size for faster training
            image_data = np.array(image)  # Convert to numpy array

            # Ensure the image is in RGB (3 channels) if it's grayscale
            if len(image_data.shape) == 2:  # If grayscale, convert to RGB
                image_data = np.stack([image_data] * 3, axis=-1)

            print(f"Loaded image: {file_path}")  # Debug print
            return image_data, label
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None, None

# Function to load images into a DataFrame using parallel processing
def load_images_to_dataframe(image_dirs):
    images = []
    labels = []

    # Use ThreadPoolExecutor to load images in parallel
    with ThreadPoolExecutor() as executor:
        futures = []

        for image_directory in image_dirs:
            label = 0 if 'benign' in image_directory.lower() else 1  # Label 0 for benign, 1 for malignant

            for subdir, dirs, files in os.walk(image_directory):
                print(f"Checking directory: {subdir}")  # Debug print
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's a valid image file
                        file_path = os.path.join(subdir, file)
                        print(f"Found image: {file_path}")  # Debug print
                        futures.append(executor.submit(load_and_process_image, file_path, label))

        # Collect the results
        for future in futures:
            image_data, label = future.result()
            if image_data is not None:
                images.append(image_data)
                labels.append(label)

    # Convert lists to DataFrame
    df = pd.DataFrame({
        'image': images,
        'label': labels
    })

    return df

# Load images into DataFrame
df = load_images_to_dataframe(image_dirs)

# Check if the DataFrame has data before proceeding
if df.empty:
    print("No images loaded. Please check the directory paths and formats.")
    exit()

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Flatten the images to 1D array for model input (since RandomForest requires 1D data)
images = np.array([image.flatten() for image in df['image']])
labels = np.array(df['label'])

# Check if images were successfully loaded
if images.shape[0] == 0:
    print("No images found after processing. Please check the image loading process.")
    exit()

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model with fewer estimators
model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1)  # Reduced number of trees

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix using seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Optionally, plot some of the images in the test set for visual inspection
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    axes[i].imshow(X_test[i].reshape(128, 128, 3))  # Reshape for visualization
    axes[i].set_title('Benign' if y_test[i] == 0 else 'Malignant')
    axes[i].axis('off')
plt.show()
