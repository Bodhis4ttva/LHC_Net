import shutil
import os
from PIL import Image
exec(open("Lib/Utils.py").read())
try:
    shutil.rmtree('Data_Images/Training')
except:
    None

os.mkdir('Data_Images/Training')
os.mkdir('Data_Images/Training/Anger')
os.mkdir('Data_Images/Training/Disgust')
os.mkdir('Data_Images/Training/Fear')
os.mkdir('Data_Images/Training/Happiness')
os.mkdir('Data_Images/Training/Sadness')
os.mkdir('Data_Images/Training/Surprise')
os.mkdir('Data_Images/Training/Neutral')

path_train = "Data/data_train.csv"
training_images, training_labels = get_data(path_train)
training_images = tf.image.resize(images=training_images, size=(224, 224), method='bilinear').numpy()
training_imagesRGB = np.zeros(shape=(28709, 224, 224, 3), dtype='float32')
for i in range(training_images.shape[0]):
    training_imagesRGB[i, :, :, :] = tf.image.grayscale_to_rgb(tf.convert_to_tensor(training_images[i, :, :, :])).numpy()


Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

for k in range(len(Categories)):
    a = training_imagesRGB[training_labels[:, k] == 1, :, :, :]
    for i in range(a.shape[0]):
        b = a[i, :, :, :].astype('uint8')
        im = Image.fromarray(b, mode='RGB')
        im.save("Data_Images/Training/"+Categories[k]+"/Images"+str(i)+".jpeg", subsampling=0, quality=100)
