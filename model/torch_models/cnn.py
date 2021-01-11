import torch
import torch.nn as nn
import torch.nn.functional as F

##https://github.com/RaffaeleGalliera/pytorch-cnn-text-classification/blob/master/model.py

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, hidden_dims):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The in_channels argument is the number of "channels" in your image going into the convolutional layer.
        # In actual images this is usually 3 (one channel for each of the red, blue and green channels),
        # however when using text we only have a single channel, t
        # he text itself. The out_channels is the number of filters and the kernel_size is the size of the filters.
        # Each of our kernel_sizes is going to be [n x emb_dim] where $n$ is the size of the n-grams.
        layers =[ nn.Conv2d(in_channels=1,out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes ]
        self.convs = nn.ModuleList(layers)

        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        # In PyTorch, RNNs want the input with the batch dimension second, whereas CNNs want the batch dimension first
        # - we do not have to permute the data here as we have already set batch_first = True in our TEXT field.
        # We then pass the sentence through an embedding layer to get our embeddings.
        # The second dimension of the input into a nn. Conv2d layer must be the channel dimension.
        # As text technically does not have a channel dimension,
        # we unsqueeze our tensor to create one.
        # This matches with our in_channels=1 in the initialization of our convolutional layers.

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        ret = self.fc1(cat)
        ret = self.fc2(ret)
        # cat = [batch size, n_filters * len(filter_sizes)]

        return ret

    @staticmethod
    def __count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)