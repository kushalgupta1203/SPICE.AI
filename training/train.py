from image_generator import train_data, val_data
from model import create_model

model = create_model()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  # Increase epochs
    batch_size=32,
    verbose=1
)

model.save("model/model.h5")
print("Model saved successfully!")
