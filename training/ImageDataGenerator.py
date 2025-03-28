import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths
dataset_path = r"D:\Projects\SPICE.AI\dataset\v0.1"

# Apply data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting dataset
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
