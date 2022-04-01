from contextlib import redirect_stdout
from itertools import product
from model.cnn import CNN
from model.cnn2 import CNN2
from model.cnn3 import CNN3
from main import execute

# Just a simple utility to run several executions of the same model with different hyperparameters

batch_sizes = [1000, 100, 10]
lrs = [0.1, 0.5]
epochss = [10]
patiences = [5]
activations = [0.2, 0.3, 0.4]
weight_decays = [0.1, 0.5]
transformms = [False, True]
dropouts = [False, True]

with open('results_cnn_1.txt', 'a') as f:
    with redirect_stdout(f):
        for b, l, e, p, a, w, t, d in product(batch_sizes,
                    lrs, epochss, patiences, activations, 
                    weight_decays, transformms, dropouts):
            print('BATCH SIZE:', b, '\nLR:', l, '\nEPOCHS:', 
                e, '\nPATIENCE:', p, '\nACTIVATION:', a, 
                '\nWEIGHT DECAY:', w, '\nTRANSFORM:', t, 
                '\nDROPOUT:', d)
            print('Test precision\tTest recall\tTest f1')
            execute(b, b, l, e, p, a, w, t, d, CNN(dropout=d))
            print('\n')
            f.flush()
    f.close()

with open('results_cnn_2.txt', 'a') as f:
    with redirect_stdout(f):
        for b, l, e, p, a, w, t, d in product(batch_sizes,
                    lrs, epochss, patiences, activations, 
                    weight_decays, transformms, dropouts):
            print('BATCH SIZE:', b, '\nLR:', l, '\nEPOCHS:', 
                e, '\nPATIENCE:', p, '\nACTIVATION:', a, 
                '\nWEIGHT DECAY:', w, '\nTRANSFORM:', t, 
                '\nDROPOUT:', d)
            print('Test precision\tTest recall\tTest f1')
            execute(b, b, l, e, p, a, w, t, d, CNN2(dropout=d))
            print('\n')
            f.flush()
    f.close()

with open('results_cnn_3.txt', 'a') as f:
    with redirect_stdout(f):
        for b, l, e, p, a, w, t, d in product(batch_sizes,
                    lrs, epochss, patiences, activations, 
                    weight_decays, transformms, dropouts):
            print('BATCH SIZE:', b, '\nLR:', l, '\nEPOCHS:', 
                e, '\nPATIENCE:', p, '\nACTIVATION:', a, 
                '\nWEIGHT DECAY:', w, '\nTRANSFORM:', t, 
                '\nDROPOUT:', d)
            print('Test precision\tTest recall\tTest f1')
            execute(b, b, l, e, p, a, w, t, d, CNN3(dropout=d))
            print('\n')
            f.flush()
    f.close()