import requests
import os
import shutil
try:
    shutil.rmtree('Data')
except:
    None

os.mkdir('Data')

try:
    os.remove('Data/data_train.csv')
except:
    None

try:
    os.remove('Data/data_val.csv')
except:
    None

try:
    os.remove('Data/data_test.csv')
except:
    None

url = 'https://www.dropbox.com/s/d939ucy3bumt9hb/data_train.csv?dl=1'
r = requests.get(url, allow_redirects=True)
open('Data/data_train.csv', 'wb').write(r.content)

url = 'https://www.dropbox.com/s/6ghexq1uijis473/data_val.csv?dl=1'
r = requests.get(url, allow_redirects=True)
open('Data/data_val.csv', 'wb').write(r.content)

url = 'https://www.dropbox.com/s/py8um99gnjjwcub/data_test.csv?dl=1'
r = requests.get(url, allow_redirects=True)
open('Data/data_test.csv', 'wb').write(r.content)




