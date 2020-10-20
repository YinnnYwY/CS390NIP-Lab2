import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
# from scipy.misc import imsave, imresize
import imageio
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
tf.compat.v1.disable_eager_execution()

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "europe.jpg"           #TODO: Add this.
STYLE_IMG_PATH = "nympheas.jpg"             #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 1.0      # Beta weight.
TOTAL_WEIGHT = 1.0

TRANSFER_ROUNDS = 10


#=============================<Helper Fuctions>=================================

def deprocessImage(img):
    # Convert a tensor into a valid image
    img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    numFilters = 3
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4. * (numFilters**2) * (CONTENT_IMG_H * CONTENT_IMG_W)**2)


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    x_var = x[:,:,1:,:] - x[:,:,:-1,:]
    y_var = x[:,1:,:,:] - x[:,:-1,:,:]
    vt_loss = tf.reduce_sum(tf.abs(x_var)) + tf.reduce_sum(tf.abs(y_var))
#     vt_loss = tf.reduce_sum((x_var)**2) + tf.reduce_sum((y_var)**2)
    return vt_loss


def compute_loss(genTensor, outputDict, styleLayerNames, contentLayerName):
    # initialize total loss
    loss = 0.0
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    # add content loss
    loss += CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)
    print("   Calculating style loss.")
    # add style loss
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        loss += STYLE_WEIGHT / len(styleLayerNames) * styleLoss(styleOutput, genOutput)
    # add total variation loss
    loss += TOTAL_WEIGHT * totalLoss(genTensor)
    return loss



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
        img = np.array(Image.fromarray(img.astype(np.uint8)).resize((iw, ih)))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


class Evaluator(object):
    def __init__(self, kFunction):
        self.kFunction = kFunction
        
    def loss(self, x):
        self.loss_, self.grads_ = self.kFunction([x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))])
        return self.loss_.astype(np.float64)

    def grads(self, x):
        return self.grads_.flatten().astype(np.float64)
    
'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    loss = compute_loss(genTensor, outputDict, styleLayerNames, contentLayerName)
    
    # Setup gradients or use K.gradients().
    grads = K.gradients(loss, genTensor)
    kFunction = K.function([genTensor], [loss] + grads)
    evaluator = Evaluator(kFunction)
    print("   Beginning transfer.")
    x = tData
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        # perform gradient descent using fmin_l_bfgs_b.
        x, tLoss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxiter=20)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(x.copy())
        saveFile = CONTENT_IMG_PATH[:-4] + STYLE_IMG_PATH[:-4] + str(i) + ".jpg"
        imageio.imwrite(saveFile, img)
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
