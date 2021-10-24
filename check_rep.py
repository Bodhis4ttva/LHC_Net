exec(open("Lib/Utils.py").read())
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(5678)
random.seed(9101112)
np.random.seed(131415)

print(tf.__version__)

path_val = "Data/data_val.csv"
validation_imagesRGB, validation_labels = etl_data(path_val)
Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

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


history = model.fit(
    train_generator,
    epochs=1,
    verbose=1,
    validation_data=(validation_imagesRGB, validation_labels)
)
pred_val = model.predict(validation_imagesRGB)
val_check = tf.keras.losses.CategoricalCrossentropy()(validation_labels, pred_val).numpy()
if val_check == 1.081964015960693359375:
    print('Replicable Results!')
else:
    print('Not Replicable Results. Change your GPU!')


# RESET
for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element