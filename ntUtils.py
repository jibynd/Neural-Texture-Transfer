def imp():
    import importlib
    tf = importlib.import_module('tensorflow')
    np = importlib.import_module('numpy')
    return tf, np


def turn_bw(img):
    tf, np = imp()
    img = tf.image.rgb_to_grayscale(img)
    img_bw = tf.image.grayscale_to_rgb(img)
    return

def get_model(texture_layers, content_layers):
    from tensorflow.python.keras import models
    tf, np = imp()
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers
    texture_outputs = [vgg.get_layer(name).output for name in texture_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = texture_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)

def deprocess_img(processed_img):
    tf, np = imp()
    x = processed_img.copy()
    # perform the  deprocessiing
    x[:, :, :, 0] += 103.939
    x[:, :, :, 1] += 116.779
    x[:, :, :, 2] += 123.68
    x = x[:, :, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def turn_var_bw(processed_var):
    tf, np = imp()
    x = processed_var
    # perform the  deprocessiing
    x = x + tf.constant([103.939, 116.779, 123.68])
    x = x[:, :, :, ::-1]
    x = tf.clip_by_value(x, 0, 255)
    x = turn_bw(x)
    x = tf.keras.applications.vgg19.preprocess_input(x)
    x = tf.dtypes.cast(x,'float32')
    return x

def load_img(img):
    from tensorflow.python.keras.preprocessing import image as k_process
    kp_image = k_process.img_to_array
    tf, np = imp()
    from PIL import Image
    max_dim = 512
    long = max([int(d) for d in img.shape])
    b, h, w, c = [int(d) for d in img.shape]
    scale = max_dim/long
    hr , wr = round(h*scale), round(w*scale)
    ar = kp_image ; pics = Image.fromarray
    img = tf.image.resize_images(img, (hr, wr))
    return img

def load_and_process_img(img):
    tf, np = imp()
    img = load_img(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img
