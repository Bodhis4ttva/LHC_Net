exec(open("Lib/Utils.py").read())

os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)




# DATA IMPORT
path_test = "Data/data_test.csv"
testing_images, testing_labels = get_data(path_test)
testing_images = tf.image.resize(images=testing_images, size=(224, 224), method='bilinear').numpy()
Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

testing_imagesRGB = np.zeros(shape=(3589, 224, 224, 3))
for i in range(testing_images.shape[0]):
    testing_imagesRGB[i, :, :, :] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(testing_images[i, :, :, :])).numpy()

del testing_images
gc.collect()

testing_imagesRGB = testing_imagesRGB.astype('float32')
testing_labels = testing_labels.astype('float32')






# MODEL IMPORT
exec(open("Lib/LHC_Net_Controller.py").read())
Params = {'num_heads': [8, 8, 7, 7, 1],
          'att_embed_dim': [196, 196, 56, 14, 25],
          'pool_size': [3, 3, 3, 3, 3],
          'norm_c': [1, 1, 1, 1, 1]}
init = [0, 0, 0, -1, -0.5]
model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params, controller_init=init)
x0 = testing_imagesRGB[0:10, :]
y0 = model(x0)
model.load_weights('Models/LHC_Net_Controller/LHC_Net_Controller')


# TTA
tf.config.run_functions_eagerly(True)
tta_pred_test = TTA_Inference(model, testing_imagesRGB)
tf.config.run_functions_eagerly(False)







# METRICS
full_train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None
)

full_train_generator = full_train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=28709,
    shuffle=False
)

h = full_train_generator.next()
training_imagesRGB = h[0]
training_labels = h[1]

del h
gc.collect()

print("LHC_NetC Perf:")
pred_train = model.predict(training_imagesRGB)
loss_train = tf.keras.losses.CategoricalCrossentropy()(training_labels, pred_train).numpy()
print('Train Loss: ', '%.17f' % loss_train)
perf_train = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(training_labels, pred_train).numpy()
print('Train Perf: ', '%.17f' % perf_train)
pred_test = model.predict(testing_imagesRGB)
perf_test = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(testing_labels, pred_test).numpy()
print('Test Perf: ', '%.17f' % perf_test)
tta_perf_test = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(testing_labels, tta_pred_test).numpy()
print('TTA Test Perf: ', '%.17f' % tta_perf_test)

print("")

pred_test_uniqueness = Check_Unique(pred_test)
tta_pred_test_uniqueness = Check_Unique(tta_pred_test)

print('Test Pred Repeated: ', pred_test_uniqueness)
print('TTA Test Pred Repeated: ', tta_pred_test_uniqueness)
