import numpy as np
import torch
class beam_search():
    def __init__(self, encoder, decoder, max_length, beam_size, attention = False):
        """
        Args:
            encoder: the encoder network
            decoder: the decoder network
            attention: boolean. True if using attention
            max_length: int. max sentence length produced
            beam_size: int.
        """    
        super(beam_search, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.max_length = max_length
        self.beam_size = beam_size
        
        
    def search(self, encoder_output, decoder_input, decoder_hidden, decoder_cell_state):
    """
    Args:
        encoder_output: output of encoder, used for attention. shape: 1 x 1 x hidden_size
        decoder_input: SOS token (e.g. torch.tensor([[SOS_token]], device=device))
        decoder_hidden: last encoder hidden vector. 
        decoder_cell_state: last encoder cell state.
    """
    decoder_input_cand = {}
    decoder_output_cand = {}
    decoder_hidden_cand = {}
    decoder_cell_state_cand = {}
    decoded_words_cand = {k:[] for k in range(beam_size)}
    final_sent = []
    final_score = []
    
    ## INIT
    if self.attention == True:
        decoder_attn = torch.zeros(max_length, max_length)
      
        decoder_output, decoder_attn, decoder_hidden, decoder_cell_state = decoder(decoder_hidden, decoder_cell_state, decoder_input, encoder_outputs)
    else: 
        decoder_output, decoder_hidden, decoder_cell_state = decoder(decoder_hidden, decoder_cell_state, decoder_input)
        
    topv, topi = decoder_output.data.topk(beam_size)
    for i in range(beam_size):
        decoded_words_cand[i].append(output_lang.index2word[topi.squeeze()[i].item()])
        decoder_input_cand[i] = topi.squeeze()[i].detach()
        decoder_hidden_cand[i] = decoder_hidden
        decoder_cell_state_cand[i] = decoder_cell_state
        
    ## BEAM-SEARCH
    word_cnt = 0
    while (bool(decoder_hidden_cand)) & (word_cnt <= max_length):
        word_cnt += 1
        topi = {}
        avail_keys = list(decoder_hidden_cand.keys())
        for b in avail_keys:
            if self.attention == True:
                decoder_output_cand[b], decoder_attn, decoder_hidden_cand[b], decoder_cell_state_canb[b] = decoder(decoder_hidden_cand[b], decoder_cell_state[b], decoder_input_cand[b],  encoder_outputs)
            else:
                decoder_output_cand[b], decoder_hidden_cand[b], decoder_cell_state_cand[b] = decoder(decoder_hidden_cand[b], decoder_cell_state_cand[b], decoder_input_cand[b])
            
            topv, topi[b] = decoder_output_cand[b].data.topk(beam_size)

            max_cand = score_all.argsort()[-beam_size:][::-1]
            decoded_sent_score = score_all[max_cand]
            #print(topv, topi[b], decoder_output_cand[b])

        cand_sentences = {}
        cand_hiddens = {}
        cand_cell_states = {}
        keys_to_rm = []
        for j in range(len(max_cand)):
            prev_cand_id = avail_keys[int(np.floor(max_cand[j]/beam_size))]

            next_id = topi[prev_cand_id].squeeze()[max_cand[j] % beam_size]
            s_cand = decoded_words_cand[prev_cand_id].copy()
            s_cand.append(output_lang.index2word[next_id.item()])
            cand_sentences[j] = s_cand
            h_cand = decoder_hidden_cand[prev_cand_id]
            cand_hiddens[j] = h_cand
            decoder_input_cand[j] = next_id.detach()   
            c_cand = decoder_cell_state_cand[prev_cand_id]
            cand_cell_states[j] = c_cand
            
        decoded_words_cand = cand_sentences
        decoder_hidden_cand = cand_hiddens
        decoder_cell_state_cand = cand_cell_states
        
        for key, s in decoded_words_cand.items():
            if 'EOS' in s:
                final_sent.append(s)
                final_score.append(decoded_sent_score[key])
                keys_to_rm.append(key)
                
        for k in keys_to_rm:
            decoder_hidden_cand.pop(k)
            decoded_words_cand.pop(k)
            decoder_cell_state_cand.pop(k)
            
    return final_sent, final_score
