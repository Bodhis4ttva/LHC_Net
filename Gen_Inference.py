# MODEL IMPORT
import os
if not os.path.isdir('Downloaded_Models'):
    exec(open("Lib/Utils.py").read())
    import requests

    os.mkdir('Downloaded_Models')
    os.mkdir('Downloaded_Models/LHC_Net')

    url = 'https://www.dropbox.com/s/gk7e7wjp9fyoea1/checkpoint?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_Net/checkpoint', 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/snartpm0ikntquy/LHC_Net.data-00000-of-00001?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_Net/LHC_Net.data-00000-of-00001', 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/tq561bgbqxdf5jh/LHC_Net.index?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_Net/LHC_Net.index', 'wb').write(r.content)

    # RESET
    for element in dir():
        if element[0:2] != "__":
            del globals()[element]
    del element

exec(open("Lib/Utils.py").read())
exec(open("Lib/LHC_Net.py").read())
Params = {'num_heads': [8, 8, 7, 7, 1],
          'att_embed_dim': [196, 196, 56, 14, 25],
          'kernel_size': [3, 3, 3, 3, 3],
          'pool_size': [3, 3, 3, 3, 3],
          'norm_c': [1, 1, 1, 1, 1]}
model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params)
x0 = np.ones(shape=(10, 224, 224, 3), dtype='float32')
y0 = model(x0)
model.load_weights('Downloaded_Models/LHC_Net/LHC_Net')

os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)



# INPUT PIPEPLINE
# the input x must be a numpy array with the following requisites:
# 1. greyscale image with shape [batch_size, 224, 224, 3]
# 2. 3 channels (the same greyscale channel is repeated in all 3 channels, all 3 channels are equal)
# 3. bilinear method is recommended to resize images to the target shape (224, 224)
# 4. every pixel in the range [0, 255]
# 5. an average pixel intensity of 129 is recommended. Consider adjusting image brightness.

# x = ...


# STANDARD INFERENCE (uncomment line 60 to produce the inference)
# pred = model.predict(x)

# TTA INFERENCE (not recommended. Too slow for real time classification)
# tf.config.run_functions_eagerly(True)
# tta_pred = TTA_Inference(model, x)
# tf.config.run_functions_eagerly(False)
