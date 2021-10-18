exec(open("Donwload_Data.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("ETL.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("ResNet34_Train.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("LHC_Train.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("Controller_Train.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("LHC_Eval.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("Controller_Eval.py").read())