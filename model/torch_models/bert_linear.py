from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, PreTrainedModel, BertConfig, load_tf_weights_in_bert
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput
import torch
from torch import nn
import torch.nn.functional as F
from model.bert_model_utils import BertPreTrainedModel


class Linear_Over_BERT(nn.Module):

    def __init__(self, bert, nhid, output_dim, dropout  ):
        super(Linear_Over_BERT, self).__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(embedding_dim, nhid)
        # dense layer 2 (Output layer)
        self.out = nn.Linear(nhid, output_dim)

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

        ret = self.relu(self.fc1(pooled_output))
        ret = self.dropout(ret)
        ret = self.out(ret)
        return ret, bert_outputs
