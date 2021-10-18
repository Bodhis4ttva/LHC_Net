exec(open("Lib/Utils.py").read())

os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


path_val = "Data/data_val.csv"
validation_imagesRGB, validation_labels = etl_data(path_val)
Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

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

opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               name='Adam')

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=1, restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/LHC_Net_Controller/LHC_Net_Controller', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)

exec(open("Lib/LHC_Net.py").read())
Params = {'num_heads': [8, 8, 7, 7, 1],
          'att_embed_dim': [196, 196, 56, 14, 25],
          'pool_size': [3, 3, 3, 3, 3],
          'norm_c': [1, 1, 1, 1, 1]}
base_model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params)
x0 = np.ones(shape=(10, 224, 224, 3), dtype='float32')
y0 = base_model(x0)
base_model.load_weights('Models/LHC_Net/LHC_Net')

exec(open("Lib/LHC_Net_Controller.py").read())
init = [0, 0, 0, -1, -0.5]
model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params, controller_init=init)
x0 = np.ones(shape=(10, 224, 224, 3), dtype='float32')
y0 = model(x0)

model.import_weights_from_lhc(base_model)


del base_model
gc.collect()


model.freeze_lhc()

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)

print(model.controller1.w.numpy())
print(model.controller2.w.numpy())
print(model.controller3.w.numpy())
print(model.controller4.w.numpy())
print(model.controller5.w.numpy())


