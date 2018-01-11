# Author: Patrick Damery
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers

# Prepare the Image Data Generator.
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    "./datasets/train/",
    target_size=(60, 100),
    batch_size=20,
)
test_generator = test_datagen.flow_from_directory(
    "./datasets/test/",
    target_size=(60, 100),
    batch_size=20,
)

# Define input tensor.
input_t = Input(shape=(60,100,3))

# Use a CNN to extract features.
conv_1 = Conv2D(filters=30, kernel_size=(6,6), activation="relu")(input_t)
# Normalize the data.
batch_norm_1 = BatchNormalization()(conv_1)
# CNN for more feature detection.
conv_2 =Conv2D(filters=60, kernel_size=(4,4), activation="relu")(batch_norm_1)
# Downscale spatial dimensions.
pool_1 = MaxPooling2D(pool_size=(2,2))(conv_2)
# Dropout to avoid overfitting.
drop_1 = Dropout(0.25)(pool_1)
# Flatten for Dense layer.
flattened_1 = Flatten()(drop_1)
# Use Dense Layer for linear operation.
dense_1 = Dense(80, activation="relu")(flattened_1)
# Dropout to avoid overfitting.
drop_2 = Dropout(0.5)(dense_1)
# Prediction layer, 2 neurons for two possible outcomes either text or not text.
prediction = Dense(2, activation="softmax")(drop_2)

# Now put it all together and create the model.
model = Model(inputs=input_t, outputs=prediction)

# Use adadelta optimizer to increase training speed.
adadelta = optimizers.Adadelta()
model.compile(optimizer=adadelta, loss="categorical_crossentropy", metrics=["accuracy"])

# Prepare Tensorboard and Checkpoint callback and start training.
tensorboard = TensorBoard(log_dir="./graph", histogram_freq=0, write_graph=True,
                        write_grads=True, write_images=True)
checkpoint = ModelCheckpoint("./checkpoints/text_detector_e_{epoch:02d}_acc_{val_acc:.2f}.hdf5",
                            verbose=1, monitor="val_acc", save_best_only=True, mode="max")
# Train model.
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=40,
    validation_data=test_generator,
    validation_steps=1000,
    callbacks=[tensorboard, checkpoint],
)

# Save trained model.
model.save("text_detector.hdf5")
label_map = (test_generator.class_indices)
