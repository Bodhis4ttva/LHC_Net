import shutil
import os
from PIL import Image
exec(open("Lib/Utils.py").read())
try:
    shutil.rmtree('Data_Images')
except:
    None

os.mkdir('Data_Images')
os.mkdir('Data_Images/Training')
os.mkdir('Data_Images/Training/Anger')
os.mkdir('Data_Images/Training/Disgust')
os.mkdir('Data_Images/Training/Fear')
os.mkdir('Data_Images/Training/Happiness')
os.mkdir('Data_Images/Training/Sadness')
os.mkdir('Data_Images/Training/Surprise')
os.mkdir('Data_Images/Training/Neutral')

path_train = "Data/data_train.csv"
training_imagesRGB, training_labels = etl_data(path_train)

Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

training_imagesRGB_trunc = np.floor(training_imagesRGB)
for k in range(len(Categories)):
    a = training_imagesRGB_trunc[training_labels[:, k] == 1, :, :, :]
    for i in range(a.shape[0]):
        b = a[i, :, :, :].astype('uint8')
        im = Image.fromarray(b, mode='RGB')
        im.save("Data_Images/Training/"+Categories[k]+"/Images"+str(i)+".jpeg", subsampling=0, quality=100)
