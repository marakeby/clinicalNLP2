
import torch
from torch import nn
import logging
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence




class RNN_Over_BERT(nn.Module):

    def __init__(self, bert, nhid,n_layers,  output_dim, bidirectional, dropout):
        super(RNN_Over_BERT, self).__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']
#         self.rnn = nn.GRU(embedding_dim,
#                           nhid,
#                           num_layers=n_layers,
#                           bidirectional=bidirectional,
#                           batch_first=True,
#                           dropout=0 if n_layers < 2 else dropout)

        self.rnn = nn.LSTM(embedding_dim,
                          nhid,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
            
        
#         self.out = nn.Linear(nhid * 2 if bidirectional else nhid, output_dim)

        self.fc1 = nn.Linear(nhid * 2 if bidirectional else nhid, nhid)
        self.fc2 = nn.Linear(nhid, output_dim)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        count_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info('RNN count parameters {}'.format(count_parameters))


    def attention_net(self, lstm_output, final_state): 
        '''
        https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
            Arguments
            ---------

            lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
            final_state : Final time-step hidden state (h_n) of the LSTM

            ---------

            Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                      new hidden state.

            Tensor Size :
                        hidden.size() = (batch_size, hidden_size)
                        attn_weights.size() = (batch_size, num_seq)
                        soft_attn_weights.size() = (batch_size, num_seq)
                        new_hidden_state.size() = (batch_size, hidden_size)

        '''
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state
    
    
    # define the forward pass
    def forward(self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict):
        # pass the inputs to the model

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        with torch.no_grad():
            pooled_output = bert_outputs[1]
            sequence_output = bert_outputs[0]

#         _, hidden = self.rnn(sequence_output)

            
        output, (hidden, final_cell_state) = self.rnn(sequence_output)
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        print ('hidden size', hidden.size())
    
#         output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)

#         attn_output = self.attention_net(output, hidden)
        
#         rel = self.relu(hidden)
        dense1 = self.fc1(hidden)
        drop = self.dropout(dense1)
        output = self.fc2(drop)
        
#         ret = self.fc1(hidden)
#         output = self.fc2(ret)
        
#         output = self.out(hidden)

      

        return output, bert_outputs
