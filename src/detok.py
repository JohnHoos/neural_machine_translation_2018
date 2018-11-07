import config
import numpy as np
def detok(ind_input, ind2word):
    '''
    Turn indices back to string
    
    Arg:
    ind_input: Each column is a sentence, element value is word indices. 
               shape = (sequence length * batch size)
    in2word: a numpy array that match ind back to word. len = # of vocab
    '''
    
    detok_output = []
    ind2word = np.array(ind2word)
    for i in range(ind_input.shape[1]):
        tok_trg = ind_input[:,i]
        tok_trg = tok_trg[tok_trg!=config.PAD_TOKEN] #remove padding
        #print(tok_trg, detok_output)
        detok_str = " ".join(ind2word[tok_trg])
        detok_output.append(detok_str)
    return detok_output
        