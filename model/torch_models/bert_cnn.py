from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, PreTrainedModel, BertConfig, load_tf_weights_in_bert
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput
import torch
from torch import nn
import torch.nn.functional as F
from model.bert_model_utils import BertPreTrainedModel


class CNN_Over_BERT(nn.Module):

    def __init__(self, bert, nhid, output_dim, nfilters, filter_sizes, dropout  ):
        super(CNN_Over_BERT, self).__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # filter_sizes=[3]
        # layers = [nn.Conv2d(in_channels=1, out_channels=nfilters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes]
        # layers =[ nn.Conv2d(in_channels=1,out_channels=nfilters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes ]
        layers = [ nn.Conv1d( in_channels=embedding_dim, out_channels=nfilters, kernel_size=fs) for fs in filter_sizes ]
        self.convs = nn.ModuleList(layers)

        # self.convs = nn.ModuleList(layers)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(len(filter_sizes) * nfilters, nhid)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(nhid, output_dim)
        # softmax activation function
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

        pooled_output = bert_outputs[1]
        sequence_output = bert_outputs[0]
        embedded = sequence_output.permute(0,2,1)
        # print(sequence_output.size())
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved = [F.relu(conv(sequence_output)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim=1))

        ret = self.fc1(cat)
        ret = self.fc2(ret)

        # conv= nn.Conv1d()
        # cnn1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1)
        # self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        #
        # x= cnn1d_1(sequence_output)
        # x = self.fc1(pooled_output)
        # x = self.relu(x)
        # x = self.dropout(x)
        # # output layer
        # x = self.fc2(x)

        # apply softmax activation
        # x = self.softmax(x)

        return ret, bert_outputs
