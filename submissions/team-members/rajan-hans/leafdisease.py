# Complete CNN Pipeline for Bangladeshi Crop Disease Dataset with Sub-Subfolder Class Structure



# 6. EDA - plot samples and class distribution
plt.figure(figsize=(14, 6))
for i, class_idx in enumerate(np.unique(y)):
    idxs = np.where(y == class_idx)[0][:5]
    for j, idx in enumerate(idxs):
        plt.subplot(len(all_classes), 5, i*5+j+1)
        plt.imshow(X[idx])
        plt.title(all_classes[class_idx])
        plt.axis('off')
plt.suptitle("Sample Images Per Class")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(y)
plt.title('Class Distribution')
plt.xticks(ticks=range(len(all_classes)), labels=all_classes, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 7. Preprocessing: normalize to [0,1] and standardize
X = X.astype('float32') / 255.0
mean = np.mean(X, axis=(0,1,2), keepdims=True)
std = np.std(X, axis=(0,1,2), keepdims=True)
X = (X - mean) / (std + 1e-7)

# One-hot encode labels
Y = to_categorical(y, num_classes=len(all_classes))

# 8. Split into train/val/test with stratification
X_train, X_temp, Y_train, Y_temp, y_train, y_temp = train_test_split(
    X, Y, y, test_size=0.3, stratify=y, random_state=seed
)
X_val, X_test, Y_val, Y_test, y_val, y_test = train_test_split(
    X_temp, Y_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 9. Handle class imbalance (optional but best practice)
cw = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(cw))
print("Class Weights:", class_weight_dict)

# 10. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    shear_range=0.2
)
datagen.fit(X_train)

# Visualize augmentations
aug_iter = datagen.flow(X_train, Y_train, batch_size=1)
plt.figure(figsize=(10,2))
for i in range(5):
    img, _ = next(aug_iter)
    plt.subplot(1,5,i+1)
    plt.imshow(np.clip((img[0]*std + mean).astype('float32'), 0, 1))
    plt.axis('off')
plt.suptitle("Augmented Images")
plt.show()

# 11. Build CNN model
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.35),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn(X_train.shape[1:], len(all_classes))
model.summary()

# 12. Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 13. Setup callbacks: early stopping, checkpoint, LR schedule
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_leaf_disease_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# 14. Train model
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train)//BATCH_SIZE,
    epochs=50,
    validation_data=(X_val, Y_val),
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# 15. Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# 16. Evaluate model on test set
score = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {score[0]:.4f} - Test Accuracy: {score[1]:.4f}")

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)

print(classification_report(y_true, y_pred, target_names=all_classes))
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_classes, yticklabels=all_classes, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Visualize misclassified images
misclassified_idx = np.where(y_pred != y_true)[0]
for i in range(min(5, len(misclassified_idx))):
    idx = misclassified_idx[i]
    plt.imshow((X_test[idx]*std + mean).clip(0,1))
    plt.title(f"True: {all_classes[y_true[idx]]}, Pred: {all_classes[y_pred[idx]]}")
    plt.axis('off')
    plt.show()

# 17. Save model in .keras and .h5 formats
model.save('leaf_disease_cnn_model.keras')
model.save('leaf_disease_cnn_model.h5')
print("Models saved as .keras and .h5")

# 18. (Optional) Export to TFLite for mobile/embedded use
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('leaf_disease_cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model exported.")

# 19. (Optional) Load model for inference
from tensorflow.keras.models import load_model
model_loaded = load_model('leaf_disease_cnn_model.keras')
model_loaded = load_model('leaf_disease_cnn_model.h5')
