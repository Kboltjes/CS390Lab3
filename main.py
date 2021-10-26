import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
tf.compat.v1.disable_eager_execution()

# Links Used:
# https://blog.keras.io/category/demo.html
# https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imresize.html
# https://stackoverflow.com/questions/49834380/k-gradientsloss-input-img0-return-none-keras-cnn-visualization-with-ten
# https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta/66222183
# https://stackoverflow.com/questions/44552585/prevent-tensorflow-from-accessing-the-gpu
# https://www.programcreek.com/python/example/66755/scipy.optimize.fmin_l_bfgs_b

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

REFERENCE_IMAGE_DIR = "ReferenceImages/"
CONTENT_IMG_PATH = REFERENCE_IMAGE_DIR + "PurdueGateway.jpg"
STYLE_IMG_PATH = REFERENCE_IMAGE_DIR + "starryNight.jpg"
MAX_ITER = 3000

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500
CONTENT_IMG_D = 3

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 3



#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    return img.reshape(CONTENT_IMG_H, CONTENT_IMG_W, CONTENT_IMG_D)


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    return STYLE_WEIGHT * (K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4. * (CONTENT_IMG_D**2) * ((STYLE_IMG_H * STYLE_IMG_W)**2)))


def contentLoss(content, gen):
    return CONTENT_WEIGHT * K.sum(K.square(gen - content))


def totalLoss(x):
    return None


#=========================<Pipeline Functions>==================================



def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


class Evaluator(object):
    def __init__(self, inTensor, outputs):
        self.inTensor = inTensor
        self.outputs = outputs
        self.gradient_values = None

    def loss(self, point):
        kfunc = K.function([self.inTensor], self.outputs)
        out = kfunc(point)
        loss_value, self.gradient_values = out[0], out[1]
        return loss_value

    def grads(self, point):
        return self.gradient_values


'''
TODO: Alot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData, dtype=tf.float64)
    styleTensor = K.variable(sData, dtype=tf.float64)
    flatTensor = K.placeholder(CONTENT_IMG_H * CONTENT_IMG_W * 3, name="testing1", dtype=tf.float64)
    genTensor = K.reshape(flatTensor, (1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += contentLoss(contentOutput, genOutput)
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        styleGenOutput = styleLayer[2, :, :, :]
        loss += styleLoss(styleOutput, styleGenOutput)

    grads = K.gradients(loss, flatTensor)[0]
    outputs = [loss]
    outputs.append(grads)
    evaluator = Evaluator(flatTensor, outputs)

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        x, tLoss, info = fmin_l_bfgs_b(evaluator.loss, x0=tData.flatten(), maxiter=MAX_ITER, fprime=evaluator.grads)
        print("      Loss: %f." % tLoss)

        img = deprocessImage(x)

        saveFile = f"finishedImg{i}.png"
        imsave(saveFile, img)
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")





#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()