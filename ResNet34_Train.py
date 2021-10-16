exec(open("Lib/Utils.py").read())

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(5678)
random.seed(9101112)
np.random.seed(131415)
print(tf.__version__)

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

testing_imagesRGB = np.zeros(shape=(3589, 224, 224, 3))
for i in range(testing_images.shape[0]):
    testing_imagesRGB[i, :, :, :] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(testing_images[i, :, :, :])).numpy()

del validation_images
del testing_images
gc.collect()














train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
    rotation_range=30,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=True
)

train_generator = train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=48,
)

ResNet34, preprocess_input = Classifiers.get('resnet34')
base_model = ResNet34(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.input
y = preprocess_input(x)
y = base_model(y)
y = tf.keras.layers.GlobalAveragePooling2D()(y)
y = tf.keras.layers.Dense(units=4096, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=3))(y)
y = tf.keras.layers.Dropout(rate=0.4)(y)
y = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=4))(y)
y = tf.keras.layers.Dropout(rate=0.4)(y)
y = tf.keras.layers.Dense(units=7, activation='softmax')(y)
model = tf.keras.Model(inputs=x, outputs=y)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               name='Adam')

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=30, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/Pre', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)


history = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)

pred_test = model.predict(testing_imagesRGB)
perf_test = tf.keras.metrics.CategoricalAccuracy()(testing_labels, pred_test).numpy()
print("Test Accuracy: ")
print(perf_test)
print("")
































train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0,
    zoom_range=0.1,
    horizontal_flip=True
)


train_generator = train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=64,
)


opt = tf.keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=20, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/Pre', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)

history2 = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)

pred_test = model.predict(testing_imagesRGB)
perf_test = tf.keras.metrics.CategoricalAccuracy()(testing_labels, pred_test).numpy()
print("Test Accuracy: ")
print(perf_test)
print("")
























tf.random.set_seed(1115)
random.seed(1115)
np.random.seed(1115)


train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
)


train_generator = train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=64
)



opt = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/Pre', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)



history3 = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)


pred_test = model.predict(testing_imagesRGB)

perf_test2 = tf.keras.metrics.CategoricalAccuracy()(testing_labels, pred_test).numpy()
print("Test Accuracy: ")
print(perf_test2)
print("")