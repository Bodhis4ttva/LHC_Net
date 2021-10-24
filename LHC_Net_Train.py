import os
if not os.path.isdir('Models'):
    os.mkdir('Models')

import json
with open('Settings/Training_Settings.json') as json_file1:
    training_settings = json.load(json_file1)
with open('Settings/LHC_Settings.json') as json_file2:
    lhc_settings = json.load(json_file2)

Params = lhc_settings

del lhc_settings

exec(open("Lib/Utils.py").read())
exec(open("Lib/LHC_Net.py").read())

seed = training_settings['seed']

os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(tf.__version__)

path_val = "Data/data_val.csv"
validation_imagesRGB, validation_labels = etl_data(path_val)

Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']




ResNet34, preprocess_input = Classifiers.get('resnet34')
resnet_model = ResNet34(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = resnet_model.input
y = preprocess_input(x)
y = resnet_model(y)
y = tf.keras.layers.GlobalAveragePooling2D()(y)
y = tf.keras.layers.Dense(units=4096, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=3))(y)
y = tf.keras.layers.Dropout(rate=0.4)(y)
y = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=4))(y)
y = tf.keras.layers.Dropout(rate=0.4)(y)
y = tf.keras.layers.Dense(units=7, activation='softmax')(y)
base_model = tf.keras.Model(inputs=x, outputs=y)

model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params)
x0 = np.ones(shape=(10, 224, 224, 3), dtype='float32')
y0 = model(x0)
model.import_w(base_model)

del x
del y
del base_model
del ResNet34
del preprocess_input
del resnet_model
gc.collect()


train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
    rotation_range=training_settings['rotation_range'][0],
    width_shift_range=training_settings['width_shift_range'][0],
    height_shift_range=training_settings['height_shift_range'][0],
    shear_range=training_settings['shear_range'][0],
    zoom_range=training_settings['zoom_range'][0],
    horizontal_flip=training_settings['horizontal_flip'][0]
)

train_generator = train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=training_settings['batch_size'][0],
    seed=seed+1
)


opt = tf.keras.optimizers.Adam(learning_rate=training_settings['learning_rate'][0],
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               name='Adam')

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=training_settings['patience'][0], restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/LHC_Net_Standalone/LHC_Net_Standalone', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)


history = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)


#  2.2
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
    rotation_range=training_settings['rotation_range'][1],
    width_shift_range=training_settings['width_shift_range'][1],
    height_shift_range=training_settings['height_shift_range'][1],
    shear_range=training_settings['shear_range'][1],
    zoom_range=training_settings['zoom_range'][1],
    horizontal_flip=training_settings['horizontal_flip'][1]
)


train_generator = train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=training_settings['batch_size'][1],
    seed=seed+1
)


opt = tf.keras.optimizers.SGD(learning_rate=training_settings['learning_rate'][1])

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=training_settings['patience'][1], restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/LHC_Net_Standalone/LHC_Net_Standalone', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)

history2 = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)


#  3.2
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=None,
)


train_generator = train_data_gen.flow_from_directory(
    directory='Data_Images/Training',
    target_size=(224, 224),
    color_mode='rgb',
    classes=Categories,
    class_mode='categorical',
    batch_size=training_settings['batch_size'][2],
    seed=seed+2
)



opt = tf.keras.optimizers.SGD(learning_rate=training_settings['learning_rate'][2])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callback1 = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=training_settings['patience'][2], restore_best_weights=True)
callback2 = tf.keras.callbacks.ModelCheckpoint('Models/LHC_Net_Standalone/LHC_Net_Standalone', monitor='val_categorical_accuracy', verbose=1, save_best_only=True)


history3 = model.fit(
    train_generator,
    epochs=300,
    verbose=1,
    callbacks=[callback1, callback2],
    validation_data=(validation_imagesRGB, validation_labels)
)



# RESET
for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element