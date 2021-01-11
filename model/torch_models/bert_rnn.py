
import torch
from torch import nn



class RNN_Over_BERT(nn.Module):

    def __init__(self, bert, nhid,n_layers,  output_dim, bidirectional, dropout):
        super(RNN_Over_BERT, self).__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
                          nhid,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(nhid * 2 if bidirectional else nhid, output_dim)

        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # relu activation function
        # self.relu = nn.ReLU()
        # # dense layer 1
        # self.fc1 = nn.Linear(nhid, 128)
        # # dense layer 2 (Output layer)
        # self.fc2 = nn.Linear(128, nlabels)
        # # softmax activation function
        # self.softmax = nn.LogSoftmax(dim=1)

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

        _, hidden = self.rnn(sequence_output)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)

        # # conv= nn.Conv1d()
        # cnn1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
        # self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        #
        # x= cnn1d_1(sequence_output)
        # x = self.fc1(pooled_output)
        # x = self.relu(x)
        # x = self.dropout(x)
        # # output layer
        # x = self.fc2(x)
        #
        # # apply softmax activation
        # # x = self.softmax(x)

        return output, bert_outputs
