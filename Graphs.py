import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


f = open('result.json', )
data = json.load(f)
df = pd.DataFrame(data)

for i in range(24):

    data = df[f'{i}']
    print(data)
    # plt.rc('xtick', labelsize=15)
    # plt.rc('ytick', labelsize=15)
    # plt.plot(np.mean(data.loc[['training_loss']][0], axis=1), label='Training loss')
    # plt.plot(np.mean(data.loc[['validation_loss']][0], axis=1), label='Validation loss')
    # plt.legend()
    # plt.xlabel('Epochs', size=13)
    # plt.ylabel('Loss', size=13)
    # type = data.loc[['type']][0]
    # batches = data.loc[['batches']][0]
    # epochs = data.loc[['epochs']][0]
    # Train_size = data.loc[['training_points']][0]
    # Test_size = data.loc[['testing_points']][0]
    # Val_size = data.loc[['validation_points']][0]
    # Test_accuracy = data.loc[['testing_accuracy']][0]
    #
    # plt.title(f'LOSS \n '
    #           # f'TYPE: {type},'
    #           f' BATCHES:{batches}, EPOCHS: {epochs}, TRAIN SIZE: {Train_size}, VAL SIZE: {Val_size}, \n '
    #           f'TEST SIZE: {Test_size}, TEST ACCURACY: {Test_accuracy}')
    # plt.savefig(f'result/{i}.pdf')
    # plt.show()


