import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    # mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # normalize and reshape
    X_test = X_test / 255.0
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # load model to be tested
    model = load_model("./models/conv_model.keras")

    # make confusion model using mnist data and model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Regular Model")
    plt.show()

if __name__ == "__main__":
    main()