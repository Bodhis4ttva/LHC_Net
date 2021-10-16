exec(open("Lib/Utils.py").read())

os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


path_val = "Data/data_val.csv"
path_test = "Data/data_test.csv"
validation_images, validation_labels = get_data(path_val)
testing_images, testing_labels = get_data(path_test)
validation_images = tf.image.resize(images=validation_images, size=(224, 224), method='bilinear').numpy()
testing_images = tf.image.resize(images=testing_images, size=(224, 224), method='bilinear').numpy()
Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

validation_imagesRGB = np.zeros(shape=(3589, 224, 224, 3))
for i in range(validation_images.shape[0]):
    validation_imagesRGB[i, :, :, :] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(validation_images[i, :, :, :])).numpy()

del validation_images
gc.collect()

testing_imagesRGB = np.zeros(shape=(3589, 224, 224, 3))
for i in range(testing_images.shape[0]):
    testing_imagesRGB[i, :, :, :] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(testing_images[i, :, :, :])).numpy()

del testing_images
gc.collect()

validation_imagesRGB = validation_imagesRGB.astype('float32')
testing_imagesRGB = testing_imagesRGB.astype('float32')
validation_labels = validation_labels.astype('float32')
testing_labels = testing_labels.astype('float32')



train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
)

train_generator = train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=32
)

# opt = tf.keras.optimizers.SGD(learning_rate=0.01)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               name='Adam')

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=1, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/LHC_Net_Controller/LHC_Net_Controller', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
callback3 = cb3(x=testing_imagesRGB, y=testing_labels)

exec(open("Lib/LHC_Net.py").read())
Params = {'num_heads': [8, 8, 7, 7, 1],
          'att_embed_dim': [196, 196, 56, 14, 25],
          'pool_size': [3, 3, 3, 3, 3],
          'norm_c': [1, 1, 1, 1, 1]}
base_model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params)
x0 = testing_imagesRGB[0:10, :]
y0 = base_model(x0)
base_model.load_weights('Models/LHC_Net/LHC_Net')

exec(open("Lib/LHC_Net_Controller.py").read())
init = [0, 0, 0, -1, -0.5]
model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params, controller_init=init)
x0 = testing_imagesRGB[0:10, :]
y0 = model(x0)

model.import_weights_from_lhc(base_model)


del base_model
gc.collect()


model.freeze_lhc()

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
pred_val = model.predict(validation_imagesRGB)
pred_test = model.predict(testing_imagesRGB)
perf_val = tf.keras.metrics.CategoricalAccuracy()(validation_labels, pred_val).numpy()
perf_test = tf.keras.metrics.CategoricalAccuracy()(testing_labels, pred_test).numpy()
print("Validation Accuracy: ", perf_val)
print("Test Accuracy: ", perf_test)


history = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2, callback3],
    validation_data=(validation_imagesRGB, validation_labels)
)

print(model.controller1.w.numpy())
print(model.controller2.w.numpy())
print(model.controller3.w.numpy())
print(model.controller4.w.numpy())
print(model.controller5.w.numpy())


