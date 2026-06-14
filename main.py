
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print(f"TensorFlow {tf.__version__}")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0


x_train = x_train.reshape(-1, 28, 28, 1)   # (60000, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)    # (10000, 28, 28, 1)

print(f"Train: {x_train.shape}  |  Test: {x_test.shape}")
print(f"Labels: {np.unique(y_train)}  (classes 0–9)\n")


model = tf.keras.Sequential([

    tf.keras.Input(shape=(28, 28, 1), name="input"),
    #1 : conv+pool
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",
        name="conv1"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

    # 2:Conv + pool 
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",
        name="conv2"
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

    # Flatten 
    tf.keras.layers.Flatten(name="flatten"),


    tf.keras.layers.Dense(128, activation="relu",  name="fc1"),
    tf.keras.layers.Dropout(0.3, name="dropout"),         
    tf.keras.layers.Dense(10,  activation="softmax", name="output"),  
], name="MNIST_CNN")

model.summary()


model.compile(
    optimizer="adam",                           # adaptive learning-rate SGD
    loss="sparse_categorical_crossentropy",     # integer labels (not one-hot)
    metrics=["accuracy"]
)



history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,      # 10 % of training data used for validation
    verbose=1
)

#5.  EVALUATION


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n{'─'*40}")
print(f"  Test accuracy : {test_acc * 100:.2f} %")
print(f"  Test loss     : {test_loss:.4f}")
print(f"{'─'*40}\n")




epochs_range = range(1, len(history.history["accuracy"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN Training History — MNIST", fontsize=14, fontweight="bold")

ax1 = axes[0]
ax1.plot(epochs_range, history.history["accuracy"],
         color="#2196F3", linewidth=2, marker="o", markersize=4, label="Training")
ax1.plot(epochs_range, history.history["val_accuracy"],
         color="#4CAF50", linewidth=2, marker="s", markersize=4,
         linestyle="--", label="Validation")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0.95, 1.01)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(epochs_range, history.history["loss"],
         color="#F44336", linewidth=2, marker="o", markersize=4, label="Training")
ax2.plot(epochs_range, history.history["val_loss"],
         color="#FF9800", linewidth=2, marker="s", markersize=4,
         linestyle="--", label="Validation")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved → cnn_training_curves.png")


