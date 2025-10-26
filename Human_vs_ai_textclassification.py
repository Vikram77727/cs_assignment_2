!pip install --upgrade keras-core
!pip install -q keras-nlp
!pip install seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras_nlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

print("TensorFlow version:", tf.__version__)
print("KerasNLP version:", keras_nlp.__version__)

# Load your dataset
file_path = '/content/drive/MyDrive/ai_human_content_detection_dataset.csv'
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Use the correct columns - text_content for text and label for target
text_col = 'text_content'
target_col = 'label'

print(f"\nUsing '{text_col}' as text column")
print(f"Using '{target_col}' as target column")

# Create the training dataset
df_train_essays = df[[text_col, target_col]].copy()
df_train_essays.columns = ['text', 'generated']

print(f"\nFinal training dataset shape: {df_train_essays.shape}")
print(f"Class distribution:\n{df_train_essays['generated'].value_counts()}")

# Check if we need to encode the target variable
if df_train_essays['generated'].dtype == 'object':
    print("\nEncoding categorical target variable...")
    label_encoder = LabelEncoder()
    df_train_essays['generated_encoded'] = label_encoder.fit_transform(df_train_essays['generated'])
    print("Classes mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    target_column = 'generated_encoded'
else:
    target_column = 'generated'
    print("Target variable is already numerical")

print(f"\nTarget variable unique values: {df_train_essays[target_column].unique()}")
print(f"Target variable type: {df_train_essays[target_column].dtype}")

# 1️⃣ Prepare classifier
SEQ_LENGTH = 512

preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=SEQ_LENGTH,
)

# Use correct number of classes based on your target
num_classes = len(df_train_essays[target_column].unique())
print(f"\nNumber of classes: {num_classes}")

classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=num_classes,
    activation='softmax',  # Changed from None to softmax for multi-class
    preprocessor=preprocessor,
)

# 2️⃣ Compile with TensorFlow's Keras optimizer
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Changed to False since we have softmax
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

classifier.backbone.trainable = False
classifier.summary()

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_train_essays["text"],
    df_train_essays[target_column],  # Use the correct target column
    test_size=0.33,
    random_state=42,
    stratify=df_train_essays[target_column]  # Stratify to maintain class distribution
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Class distribution in training: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Class distribution in test: {pd.Series(y_test).value_counts().to_dict()}")

# 4️⃣ Fit model
print("\nTraining the model...")
try:
    history = classifier.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        epochs=3,  # Increased to 3 epochs for better learning
        batch_size=32,  # Reduced batch size for stability
        verbose=1
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    print("\nTrying alternative approach...")
    
    # Alternative: Use smaller batch size and 1 epoch
    history = classifier.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_test, y_test),
        epochs=1,
        batch_size=16,
        verbose=1
    )

# 5️⃣ Evaluate and Calculate Accuracy
print("\nMaking predictions...")
y_pred_test = classifier.predict(X_test)
y_pred_classes = np.argmax(y_pred_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\n📊 MODEL PERFORMANCE METRICS:")
print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calculate additional metrics
print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_classes))

# Confusion Matrix
def displayConfusionMatrix(y_true, y_pred, dataset):
    plt.figure(figsize=(10, 8))
    
    accuracy = accuracy_score(y_true, np.argmax(y_pred, axis=1))
    
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        np.argmax(y_pred, axis=1),
        cmap=plt.cm.Blues,
        values_format='d'
    )
    
    disp.ax_.set_title(f"Confusion Matrix - {dataset} Dataset\nAccuracy: {accuracy:.2%}", 
                       fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

displayConfusionMatrix(y_test, y_pred_test, "Test")

# 6️⃣ Additional Visualizations
if 'history' in locals():
    plt.figure(figsize=(15, 5))
    
    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy During Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 7️⃣ Final Summary
print(f"\n🎯 FINAL MODEL SUMMARY:")
print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Model successfully classified {len(X_test)} test samples")
print(f"Number of classes: {num_classes}")

# 8️⃣ Test on sample predictions
print(f"\n🔍 SAMPLE PREDICTIONS:")
sample_texts = [
    "This is a sample text to test the model prediction capabilities.",
    "The quick brown fox jumps over the lazy dog in this example sentence."
]

for i, text in enumerate(sample_texts):
    prediction = classifier.predict([text])
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    print(f"Sample {i+1}: Predicted class {predicted_class} with confidence {confidence:.4f}")
