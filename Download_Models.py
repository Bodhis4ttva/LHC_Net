import os
if not os.path.isdir('Downloaded_Models'):
    exec(open("Lib/Utils.py").read())
    import requests

    os.mkdir('Downloaded_Models')
    os.mkdir('Downloaded_Models/LHC_Net')
    os.mkdir('Downloaded_Models/LHC_NetC')

    url = 'https://www.dropbox.com/s/gk7e7wjp9fyoea1/checkpoint?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_Net/checkpoint', 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/snartpm0ikntquy/LHC_Net.data-00000-of-00001?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_Net/LHC_Net.data-00000-of-00001', 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/tq561bgbqxdf5jh/LHC_Net.index?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_Net/LHC_Net.index', 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/gdsaj9bhjrnycz5/checkpoint?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_NetC/checkpoint', 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/6pfpmyz41pgfuce/LHC_Net_Controller.data-00000-of-00001?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_NetC/LHC_Net_Controller.data-00000-of-00001', 'wb').write(r.content)

    url = 'https://www.dropbox.com/s/fkfjfer31m9ilnk/LHC_Net_Controller.index?dl=1'
    r = requests.get(url, allow_redirects=True)
    open('Downloaded_Models/LHC_NetC/LHC_Net_Controller.index', 'wb').write(r.content)

    # RESET
    for element in dir():
        if element[0:2] != "__":
            del globals()[element]
    del element