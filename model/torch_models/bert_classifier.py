from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, PreTrainedModel, BertConfig, load_tf_weights_in_bert
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput
import torch.nn.functional as F
from model.bert_model_utils import BertPreTrainedModel
from torch import nn

class BertSequenceClassificationModel(BertPreTrainedModel):
    def __init__(self, bert, freez_bert, classifer, **classifer_params):
        self.config= bert.config
        super().__init__(self.config)
        self.bert = bert
        for name, param in self.bert.named_parameters():
            print (name)
            if 'classifier' not in name:  # classifier layer
                if freez_bert:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        config = self.bert.config
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        print('BertSequenceClassificationModel constructor !')
        self.classifier = classifer(bert, **classifer_params)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # classifier = CNN_Over_BERT(self.bert)

        logits, outputs = self.classifier (
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


        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )