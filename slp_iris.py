import csv
import math
import matplotlib.pyplot as plt

DATA_FILE = "iris_setosa_versicolor.csv"

setosa, versicolor = [], []
with open(DATA_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        features = [float(row["sepal_length"]), float(row["sepal_width"]),
                    float(row["petal_length"]), float(row["petal_width"])]
        label = 0.0 if row["species"] == "Iris-setosa" else 1.0
        (setosa if label == 0.0 else versicolor).append((features, label))

train_data = setosa[:40] + versicolor[:40]        # 80 rows
val_data   = setosa[40:50] + versicolor[40:50]    # 20 rows

LEARNING_RATE = 0.1
EPOCHS = 5
weights = {"bias": 0.5, "w1": 0.5, "w2": 0.5, "w3": 0.5, "w4": 0.5}

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def forward(x, w):
    z = w["bias"] + w["w1"] * x[0] + w["w2"] * x[1] + w["w3"] * x[2] + w["w4"] * x[3]
    g = sigmoid(z)
    return z, g

def run_epoch_training(data, w, lr):
    total_sse = 0.0
    correct = 0
    for x, target in data:
        z, g = forward(x, w)
        pred = 1 if g > 0.5 else 0
        error = g - target
        sse = error ** 2

        total_sse += sse
        if pred == target:
            correct += 1

        d_bias = 2 * error * (1 - g) * g
        d_w1 = d_bias * x[0]
        d_w2 = d_bias * x[1]
        d_w3 = d_bias * x[2]
        d_w4 = d_bias * x[3]

        w["bias"] -= lr * d_bias
        w["w1"] -= lr * d_w1
        w["w2"] -= lr * d_w2
        w["w3"] -= lr * d_w3
        w["w4"] -= lr * d_w4

    avg_loss = total_sse / len(data)
    accuracy = correct / len(data)
    return avg_loss, accuracy


def run_validation(data, w):
    total_sse = 0.0
    correct = 0
    for x, target in data:
        z, g = forward(x, w)
        pred = 1 if g > 0.5 else 0
        error = g - target
        total_sse += error ** 2
        if pred == target:
            correct += 1

    avg_loss = total_sse / len(data)
    accuracy = correct / len(data)
    return avg_loss, accuracy

train_loss_history, train_acc_history = [], []
val_loss_history, val_acc_history = [], []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch_training(train_data, weights, LEARNING_RATE)
    val_loss, val_acc = run_validation(val_data, weights)

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch}: "
          f"train_loss={train_loss:.5f} train_acc={train_acc:.3f} | "
          f"val_loss={val_loss:.5f} val_acc={val_acc:.3f}")

print("\nFinal weights:", weights)
epochs_range = list(range(1, EPOCHS + 1))

plt.figure()
plt.plot(epochs_range, train_acc_history, marker="o", label="Training Accuracy")
plt.plot(epochs_range, val_acc_history, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.xticks(epochs_range)
plt.legend()
plt.grid(True)
plt.savefig("accuracy_chart.png", dpi=150, bbox_inches="tight")

plt.figure()
plt.plot(epochs_range, train_loss_history, marker="o", label="Training Loss")
plt.plot(epochs_range, val_loss_history, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (avg SSE)")
plt.title("Loss per Epoch")
plt.xticks(epochs_range)
plt.legend()
plt.grid(True)
plt.savefig("loss_chart.png", dpi=150, bbox_inches="tight")
