"""Model loading, image preprocessing and digit prediction."""
# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Loading model
model = load_model("digit_model.h5")


# Preparing and pre-processing the image
def preprocess_img(img_path):
    """Open an image, resize it to 224x224 and scale pixels to the 0-1 range."""
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    """Return the most likely digit (0-9) for a preprocessed image."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
