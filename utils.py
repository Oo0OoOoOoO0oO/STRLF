import numpy as np

def getCov(data):
    dataCov = []
    for trial in data:
        cov = np.corrcoef(trial)
        dataCov.append(cov)

    dataCov = np.array(dataCov)
    return dataCov