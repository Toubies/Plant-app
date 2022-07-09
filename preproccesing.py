import PIL
import numpy as np
import matplotlib



def preprocess_image(original_image, size: tuple = (299, 299), RGB: bool = True,
                     scale: bool = True, dtype: str = 'int32'):
    """Preprocess image by resizing and dividing each pixel by 255. The function convert image to RGB by default."""

    if RGB:
        image = PIL.Image.open(original_image).convert('RGB')
    else:
        image = PIL.Image.open(original_image)
    image = image.resize(size)
    image = np.array(image,dtype=dtype)
    if scale:
        image = image / 255
    return image

def get_class_name(predictions: list, class_dictionary: dict):
    for prediction in predictions:
      cl_str = [x[0] for x in class_dictionary.items() if x[1] == prediction]
    return cl_str[0]

def class_int(predictions: list, class_dictionary: dict):
    for prediction in predictions:
      cl_int = [x[1] for x in class_dictionary.items() if x[0] == prediction]
    return cl_int[0]

def prepare_to_predict(image):
    return np.expand_dims(image, axis=0)