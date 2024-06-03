import h5py
import tensorflow as tf

with h5py.File("data.h5", "r") as hf:
    X_train = hf["X_train"][:]
    y_train = hf["y_train"][:]
    X_valid = hf["X_valid"][:]
    y_valid = hf["y_valid"][:]
    X_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

model = tf.keras.models.load_model("models_checkpoints/Fer2013_7203.h5")
# evaluate the model on the test set
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print(acc,loss)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))