import numpy as np
import os
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#  NORMALIZATION 

def normalize_landmarks(sequence):
    
    seq = sequence.copy()
    for i in range(seq.shape[0]):
        wrist1 = seq[i, 0:3].copy()
        wrist2 = seq[i, 63:66].copy()
        seq[i, :63] -= np.tile(wrist1, 21)
        seq[i, 63:] -= np.tile(wrist2, 21)
    return seq


def scale_normalize_sequence(seq):
    
    result = seq.copy()
    for i in range(seq.shape[0]):
        frame = seq[i].reshape(-1, 3)
        norms = np.linalg.norm(frame, axis=1)
        max_dist = np.max(norms)
        if max_dist > 1e-6:
            result[i] = seq[i] / max_dist
    return result


#  LOAD DATA 

dataset_path = "dataset"

labels = ["i","you","we",
    "go","come","eat","drink","like","write","read","play",
    "water","toilet","food","bag","class","teacher",
    "want","please","yes","help","thank_you"
    ]
label_map = {label: i for i, label in enumerate(labels)}

x, y = [], []

for label in labels:
    folder = os.path.join(dataset_path, label)
    files = os.listdir(folder)
    print(f"  {label}: {len(files)} samples")

    for file in files:
        data = np.load(os.path.join(folder, file))
        x.append(data)
        y.append(label_map[label])

x = np.array(x, dtype=np.float32)
y = np.array(y)

print("\nRaw shape:", x.shape)
print("Class distribution:", Counter(y))


#  PREPROCESS 

x = np.array([normalize_landmarks(seq) for seq in x])
x = np.array([scale_normalize_sequence(seq) for seq in x])

# Check for NaN/Inf after normalization
assert not np.any(np.isnan(x)), "NaN found after normalization!"
assert not np.any(np.isinf(x)), "Inf found after normalization!"

print("After preprocessing:", x.shape)


#  TRAINING 

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
best_val_acc = 0.0
best_model = None

for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
    print(f"\n{'='*40}")
    print(f"  Fold {fold + 1}/5")
    print(f"{'='*40}")

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Class weights to handle imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Model

    
    model = Sequential([
        LSTM(96, return_sequences=True, input_shape=(30, x.shape[2])),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=True),
        Dropout(0.3),

        LSTM(32),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(len(labels), activation='softmax')
    ])



    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(f"best_fold_{fold+1}.h5", monitor='val_accuracy',
                        save_best_only=True, verbose=0)
    ]

    model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFold {fold+1} Test Accuracy: {acc:.4f}")
    accuracies.append(acc)

    # Track best model across folds
    if acc > best_val_acc:
        best_val_acc = acc
        best_model = model


#  RESULTS 

print("\n=== K-Fold Results ===")
print(f"Per-fold accuracies: {[f'{a:.4f}' for a in accuracies]}")
print(f"Mean Accuracy : {np.mean(accuracies):.4f}")
print(f"Std Accuracy  : {np.std(accuracies):.4f}")
print(f"Best Fold Acc : {best_val_acc:.4f}")


#  SAVE BEST MODEL 

best_model.save("gesture_lstm_model.h5")
np.save("label_classes.npy", np.array(labels))

print("\n Best model saved: gesture_lstm_model.h5")
print(" Labels saved    : label_classes.npy")