import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint
)
import matplotlib.pyplot as plt
from pathlib import Path

# FORCE GPU USAGE
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✓ {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Configuration
# ==============================================================================
# UPDATED: Changed from 384 to 385 to match your original model's dimensions
IMG_SIZE = 385 
# ==============================================================================
BATCH_SIZE = 64  # Increased for better GPU utilization
EPOCHS = 50
LEARNING_RATE = 1e-2
AUTOTUNE = tf.data.AUTOTUNE

# Kaggle dataset path
DATASET_PATH = "/kaggle/input/dataset/dataset"

print("TensorFlow version:", tf._version_)
print("Keras version:", keras._version_)

# Verify paths exist
print(f"\nDataset path exists: {os.path.exists(DATASET_PATH)}")

def load_and_preprocess_image(file_path, label):
    """Load and preprocess a single image - GPU optimized"""
    img = tf.io.read_file(file_path)
    # ==========================================================================
    # THIS IS THE KEY: 'channels=3' ensures images are RGB, not grayscale.
    # This was already correct in your script.
    img = tf.image.decode_jpeg(img, channels=3)
    # ==========================================================================
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment_image(image, label):
    """Apply data augmentation - runs on GPU"""
    # Random flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Random rotation (approximation using flip and transpose)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.rot90(image)
    
    return image, label

def create_tf_dataset(directory, validation_split=0.2, is_training=True):
    """Create TensorFlow dataset for GPU training"""
    
    full_path = os.path.join(directory, "full")
    non_full_path = os.path.join(directory, "non_full")
    
    # Get all image paths
    full_images = [str(p) for p in Path(full_path).glob("*.jpg")]
    non_full_images = [str(p) for p in Path(non_full_path).glob("*.jpg")]
    
    # Create labels (1 for full, 0 for non_full)
    full_labels = [1] * len(full_images)
    non_full_labels = [0] * len(non_full_images)
    
    # Combine
    all_images = full_images + non_full_images
    all_labels = full_labels + non_full_labels
    
    # Shuffle
    indices = np.random.RandomState(42).permutation(len(all_images))
    all_images = [all_images[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split train/val
    split_idx = int(len(all_images) * (1 - validation_split))
    
    if is_training:
        images = all_images[:split_idx]
        labels = all_labels[:split_idx]
    else:
        images = all_images[split_idx:]
        labels = all_labels[split_idx:]
    
    print(f"{'Training' if is_training else 'Validation'} samples: {len(images)}")
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Map loading and preprocessing (parallel)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    
    if is_training:
        # Shuffle and augment for training
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    
    # Batch and prefetch for GPU efficiency
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset, len(images)

def load_and_prepare_data():
    """Load data as TensorFlow datasets for optimal GPU usage"""
    
    print("\nCreating TensorFlow datasets for GPU training...")
    
    train_dataset, train_count = create_tf_dataset(DATASET_PATH, validation_split=0.2, is_training=True)
    val_dataset, val_count = create_tf_dataset(DATASET_PATH, validation_split=0.2, is_training=False)
    
    steps_per_epoch = train_count // BATCH_SIZE
    validation_steps = val_count // BATCH_SIZE
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def build_model():
    """Build transfer learning model with EfficientNetB3"""
    
    # Force model to GPU
    with tf.device('/GPU:0'):
        # Load pre-trained EfficientNetB3
        try:
            # ==================================================================
            # THIS IS THE FIX: input_shape uses IMG_SIZE (385) and 3 channels.
            # This now matches the data loading pipeline.
            base_model = EfficientNetB3(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
            # ==================================================================
            print("✓ Loaded EfficientNetB3 with ImageNet weights")
        except:
            print("⚠ Using model without pre-trained weights")
            base_model = EfficientNetB3(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights=None
            )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build custom head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
    
    return model, base_model

def create_callbacks():
    """Create training callbacks"""
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            '/kaggle/working/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def train_model(model, base_model, train_dataset, val_dataset, steps_per_epoch, validation_steps):
    """Train the model with fine-tuning strategy"""
    
    callbacks = create_callbacks()
    
    # Phase 1: Train with frozen base model
    print("\n" + "="*60)
    print("Phase 1: Training with frozen base model")
    print("="*60)
    
    # Mixed precision for faster GPU training
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen top layers
    print("\n" + "="*60)
    print("Phase 2: Fine-tuning with unfrozen top layers")
    print("="*60)
    base_model.trainable = True
    
    # Freeze all but last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        initial_epoch=20,
        verbose=1
    )
    
    return history_phase1, history_phase2

def plot_training_history(history1, history2):
    """Plot training and validation metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history1.history['accuracy'], label='Phase 1 Train', marker='o')
    axes[0, 0].plot(history1.history['val_accuracy'], label='Phase 1 Val', marker='s')
    axes[0, 0].plot([20 + i for i in range(len(history2.history['accuracy']))], 
                    history2.history['accuracy'], label='Phase 2 Train', marker='o')
    axes[0, 0].plot([20 + i for i in range(len(history2.history['val_accuracy']))], 
                    history2.history['val_accuracy'], label='Phase 2 Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history1.history['loss'], label='Phase 1 Train', marker='o')
    axes[0, 1].plot(history1.history['val_loss'], label='Phase 1 Val', marker='s')
    axes[0, 1].plot([20 + i for i in range(len(history2.history['loss']))], 
                    history2.history['loss'], label='Phase 2 Train', marker='o')
    axes[0, 1].plot([20 + i for i in range(len(history2.history['val_loss']))], 
                    history2.history['val_loss'], label='Phase 2 Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history1.history['precision'], label='Phase 1 Train', marker='o')
    axes[1, 0].plot(history1.history['val_precision'], label='Phase 1 Val', marker='s')
    axes[1, 0].plot([20 + i for i in range(len(history2.history['precision']))], 
                    history2.history['precision'], label='Phase 2 Train', marker='o')
    axes[1, 0].plot([20 + i for i in range(len(history2.history['val_precision']))], 
                    history2.history['val_precision'], label='Phase 2 Val', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history1.history['recall'], label='Phase 1 Train', marker='o')
    axes[1, 1].plot(history1.history['val_recall'], label='Phase 1 Val', marker='s')
    axes[1, 1].plot([20 + i for i in range(len(history2.history['recall']))], 
                    history2.history['recall'], label='Phase 2 Train', marker='o')
    axes[1, 1].plot([20 + i for i in range(len(history2.history['val_recall']))], 
                    history2.history['val_recall'], label='Phase 2 Val', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved as 'training_history.png'")
    plt.show()

def main():
    print("="*60)
    print("Full-Body Image Classification Training (Kaggle - GPU)")
    print("="*60)
    
    # Verify GPU
    print("\n🔍 Checking GPU availability...")
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print("⚠  WARNING: No GPU detected! Training will be slow.")
    else:
        print(f"✓ GPU is available and will be used!")
    
    # Load data
    print("\nLoading and preparing data...")
    train_dataset, val_dataset, steps_per_epoch, validation_steps = load_and_prepare_data()
    
    # Build model
    print("\nBuilding model with EfficientNetB3...")
    model, base_model = build_model()
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print("\nStarting training...")
    history1, history2 = train_model(model, base_model, train_dataset, val_dataset, 
                                     steps_per_epoch, validation_steps)
    
    # Plot results
    plot_training_history(history1, history2)
    
    # Save final model
    print("\nSaving final model as 'fullbody_classifier.keras'...")
    model.save('/kaggle/working/fullbody_classifier.keras')
    print("✓ Model saved successfully!")
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation:")
    print("="*60)
    eval_results = model.evaluate(val_dataset, steps=validation_steps, verbose=0)
    print(f"Val Loss:      {eval_results[0]:.4f}")
    print(f"Val Accuracy:  {eval_results[1]:.4f}")
    print(f"Val Precision: {eval_results[2]:.4f}")
    print(f"Val Recall:    {eval_results[3]:.4f}")
    
    # Calculate F1 Score
    precision = eval_results[2]
    recall = eval_results[3]
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Val F1-Score:  {f1_score:.4f}")
    print("="*60)

if _name_ == "_main_":
    main()