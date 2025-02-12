# Brain-Tumor-detection-Using-CNN-TensorFlow-


Brain Tumor Detection using CNN: TensorFlow 😎🔐
🚀 Introduction

Brain tumor detection is a crucial problem in medical imaging. This project demonstrates how deep learning, particularly Convolutional Neural Networks (CNNs), can be applied to classify brain tumors with 96% accuracy. The workflow covers data preprocessing, model building, training, evaluation, and deployment, making it a complete end-to-end solution.
📌 Step-by-Step Implementation
1️⃣ Importing Essential Libraries ✅

To implement this model, we use key Python libraries for deep learning, data handling, and visualization, such as TensorFlow/Keras, NumPy, Matplotlib, and Scikit-learn.
2️⃣ Setting Up Dataset Paths and Directories ✅

The dataset consists of four tumor categories:

    Glioma
    Meningioma
    Pituitary
    No Tumor (Healthy Brain Scans)

The dataset is divided into Training, Validation, and Testing sets for better generalization.
3️⃣ Loading and Preprocessing the Dataset ✅

Before training, images are:
✔ Resized to a fixed dimension for consistency.
✔ Normalized to scale pixel values between 0 and 1.
✔ One-hot encoded to categorize tumor types.
4️⃣ Visualizing Images for Each Tumor Type ✅

To ensure data integrity, sample images from each tumor category are displayed, helping identify any imbalances or misclassifications.
5️⃣ Defining Model Hyperparameters ✅

For training, key parameters are set:

    Image size (e.g., 150x150 pixels)
    Batch size (e.g., 32 images per batch)
    Number of epochs (e.g., 20 training cycles)

These parameters directly affect model performance and efficiency.
6️⃣ Data Augmentation and Preprocessing ✅

To prevent overfitting, data augmentation techniques are applied:
✔ Rotation – Small rotations help the model learn orientation variations.
✔ Shifting – Moves images slightly to introduce positional robustness.
✔ Zooming – Slight zoom to simulate different scanning depths.
✔ Flipping – Horizontal flips to increase diversity.

These transformations help create a more generalized model.
7️⃣ Building the CNN Model Architecture ✅

The CNN model consists of multiple layers:

    Convolutional Layers – Extract key features from images.
    Max Pooling Layers – Reduce dimensionality while retaining features.
    Flatten Layer – Converts the feature map into a single vector.
    Dense Layers – Fully connected layers to classify tumor types.
    Dropout Layer – Prevents overfitting by randomly disabling neurons.

The model is compiled using the Adam optimizer and categorical cross-entropy loss for multi-class classification.
8️⃣ Training and Validation ✅

The model is trained using the training dataset while its performance is evaluated on the validation dataset after each epoch. This helps track improvements and avoid overfitting.
9️⃣ Visualization Through Graphs ✅

To analyze model performance, graphs are plotted:
✔ Training vs. Validation Accuracy – Shows learning progress over epochs.
✔ Training vs. Validation Loss – Identifies underfitting or overfitting trends.

A well-trained model should have high accuracy with minimal loss.
🔟 Model Evaluation & Performance Metrics ✅

The trained model is tested on unseen data (test set) to measure its generalization. The final accuracy score is calculated, confirming the effectiveness of CNNs for brain tumor detection.
1️⃣1️⃣ Confusion Matrix & Classification Report ✅

To understand how well the model differentiates between tumor types:
✔ Confusion Matrix – Shows correct vs. incorrect predictions for each category.
✔ Precision, Recall, F1-Score – Evaluates overall classification performance.

This step helps identify areas for improvement.
1️⃣2️⃣ Saving the Trained Model ✅

Once training is complete, the model is saved for future use. It can be deployed in real-world applications such as medical diagnosis software or mobile apps.
🚀 Conclusion

This project successfully demonstrates the power of CNNs in brain tumor detection, achieving 96% accuracy. The model effectively classifies different tumor types and can assist medical professionals in making faster and more accurate diagnoses.
