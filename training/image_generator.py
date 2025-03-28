from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = r"D:\Projects\SPICE.AI\dataset\v0.1"

train_datagen = ImageDataGenerator(
    rescale=1.0/255,  
    rotation_range=40,  # Increased rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # Shear transformation
    zoom_range=0.3,  # Zooming
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],  # Adjust brightness
    fill_mode='nearest',
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
