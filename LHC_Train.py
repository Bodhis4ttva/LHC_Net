exec(open("Lib/Utils.py").read())
exec(open("Lib/LHC_Net.py").read())

os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


path_val = "Data/data_val.csv"
validation_imagesRGB, validation_labels = etl_data(path_val)
Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None
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

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/LHC_Net/LHC_Net', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)

base_model = tf.keras.models.load_model('Models/Pre')


Params = {'num_heads': [8, 8, 7, 7, 1],
          'att_embed_dim': [196, 196, 56, 14, 25],
          'pool_size': [3, 3, 3, 3, 3],
          'norm_c': [1, 1, 1, 1, 1]}

model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params)
x0 = np.ones(shape=(10, 224, 224, 3), dtype='float32')
y0 = model(x0)
model.import_w(base_model)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)
