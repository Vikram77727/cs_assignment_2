Here’s a structured **README** for your DistilBERT AI-human content classifier project:

---

# **AI-Human Text Content Classifier using DistilBERT**

## **Project Overview**

This project uses a **DistilBERT-based classifier** to detect whether a piece of text is **AI-generated or human-written**. The model is trained on a labeled dataset and provides metrics, visualizations, and sample predictions to evaluate its performance.

---

## **Features**

* Preprocesses text data using **DistilBERT tokenizer and preprocessor**.
* Handles **binary classification** (`AI-generated` vs `human-written`).
* Supports **training, validation, and test evaluation**.
* Visualizes:

  * **Confusion matrix**
  * **Training & validation accuracy**
  * **Training & validation loss**
* Generates **sample predictions** with confidence scores.

---

## **Requirements**

* Python ≥ 3.8
* TensorFlow 2.19.0
* KerasNLP 0.21.1
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn

Install required packages:

```bash
!pip install --upgrade keras-core
!pip install -q keras-nlp
!pip install seaborn
```

---

## **Dataset**

* CSV file containing text and label columns.
* Required columns:

  * `text_content` → The text input
  * `label` → Target class (0 = AI-generated, 1 = human-written)
* Example first 5 rows:

| text_content                  | label |
| ----------------------------- | ----- |
| "Sample AI-generated text..." | 0     |
| "Human-written essay text..." | 1     |

---

## **Usage**

1. **Load dataset**

```python
df = pd.read_csv('/path/to/ai_human_content_detection_dataset.csv')
```

2. **Preprocess data**

* Select `text_content` as input and `label` as target.
* Encode target variable if it’s categorical.

3. **Define and compile the model**

```python
preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=512
)

classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
    activation='softmax',
    preprocessor=preprocessor
)

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
```

4. **Train-test split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    df['text_content'],
    df['label'],
    test_size=0.33,
    stratify=df['label'],
    random_state=42
)
```

5. **Train the model**

```python
history = classifier.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=3,
    batch_size=32
)
```

6. **Evaluate model**

```python
y_pred = np.argmax(classifier.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))
```

7. **Visualize results**

* Confusion matrix
* Accuracy and loss plots

---

## **Sample Predictions**

```python
sample_texts = [
    "This is a sample text to test the model.",
    "The quick brown fox jumps over the lazy dog."
]

for text in sample_texts:
    pred = classifier.predict([text])
    predicted_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)
    print(f"Text: {text}")
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")
```

---

## **Performance Metrics**

* Example metrics from training:

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | 74%   |
| Validation Accuracy | 73.5% |
| Test Accuracy       | 73.5% |

---

## **Notes**

* The backbone of DistilBERT is **frozen** during training to reduce overfitting and speed up training.
* The model supports **softmax activation** for multi-class or binary classification.
* Can be extended for **larger datasets** or fine-tuned for better performance.

---

