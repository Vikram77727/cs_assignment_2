import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras_core as keras
import keras_nlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# 1️ Prepare classifier
SEQ_LENGTH = 512

preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
    "distil_bert_base_en_uncased",
    sequence_length=SEQ_LENGTH,
)

classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
    activation=None,
    preprocessor=preprocessor,
)

# 2️ Compile AFTER creating the classifier
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

classifier.backbone.trainable = False
classifier.summary()

# 3️ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df_train_essays["text"],
    df_train_essays["generated"],
    test_size=0.33,
    random_state=42
)

# 4️ Fit model
classifier.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    epochs=1,
    batch_size=64
)

# 5️ Evaluate
y_pred_test = classifier.predict(X_test)

def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        np.argmax(y_pred, axis=1),
        display_labels=["Not Generated","Generated"],
        cmap=plt.cm.Blues
    )

    tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
    f1_score = tp / (tp + ((fn + fp)/2))

    disp.ax_.set_title(f"Confusion Matrix on {dataset} Dataset -- F1 Score: {f1_score:.2f}")

displayConfusionMatrix(y_test, y_pred_test, "Test")
