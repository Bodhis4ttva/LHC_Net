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

exec(open("Download_Models.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("LHC_Downloaded_Eval.py").read())

for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element

exec(open("Controller_Downloaded_Eval.py").read())