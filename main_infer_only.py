exec(open("Lib/Utils.py").read())
import requests



import shutil
try:
    shutil.rmtree('Base_Model')
except:
    None

os.mkdir('Base_Model')
os.mkdir('Base_Model/assets')
os.mkdir('Base_Model/variables')

url = 'https://www.dropbox.com/s/wv3xpps8r7zxqg2/saved_model.pb?dl=1'
r = requests.get(url, allow_redirects=True)
open('Base_Model/saved_model.pb', 'wb').write(r.content)

url = 'https://www.dropbox.com/s/dwzmm2wy6vwo7a6/variables.data-00000-of-00001?dl=1'
r = requests.get(url, allow_redirects=True)
open('Base_Model/variables/variables.data-00000-of-00001', 'wb').write(r.content)

url = 'https://www.dropbox.com/s/blh7e4pf87oe5rr/variables.index?dl=1'
r = requests.get(url, allow_redirects=True)
open('Base_Model/variables/variables.index', 'wb').write(r.content)


os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)
print(tf.__version__)


path_val = "Data/data_val.csv"
validation_imagesRGB, validation_labels = etl_data(path_val)
path_test = "Data/data_test.csv"
testing_imagesRGB, testing_labels = etl_data(path_test)
Categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

model = tf.keras.models.load_model('Base_Model')

pred_val = model.predict(validation_imagesRGB)
perf_val = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(validation_labels, pred_val).numpy()
print('Val Perf: ', '%.17f' % perf_val)
pred_test = model.predict(testing_imagesRGB)
perf_test = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(testing_labels, pred_test).numpy()
print('Test Perf: ', '%.17f' % perf_test)