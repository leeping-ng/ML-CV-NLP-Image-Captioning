import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # remove the last fully connected layer
        # resnet.children() is a generator, here, a list of all layers is created
        modules = list(resnet.children())[:-1]
        # create a resnet model without the fc layer
        self.resnet = nn.Sequential(*modules)
        # resnet.fc returns Linear(in_features=2048, out_features=1000, bias=True)
        # here, creating a linear layer with in_features=2048 and out_features=256
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        # returns torch.Size([batch_size, 2048, 1, 1])
        features = self.resnet(images)
        # reshape to torch.Size([batch_size, 2048])
        features = features.view(features.size(0), -1)
        # Features shape: torch.Size([batch_size, embed_size])
        features = self.embed(features)
        
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # initialise the hidden and cell states (h_0, c_0)
        self.hidden = (torch.randn(self.num_layers, 1, self.hidden_size), #h_0
                torch.randn(self.num_layers, 1, self.hidden_size)) #c_0

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # batch_first=True causes input and output tensors to be of (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, \
            num_layers=num_layers, batch_first=True)

        # linear layer that maps hidden state output dim to vocab size
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):

        # Discard the <end> word to avoid predicting when <end> is the RNN input
        captions = captions[:, :-1]
        
        # returns torch.Size([batch_size, caption_length, embed_size])
        embeds = self.word_embeddings(captions)

        # unsqueeze dim=1 to be able to concat, as embeds has caption_length
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)

        lstm_out, _ = self.lstm(inputs)

        # return pytorch tensor with size [batch_size. captions.shape[1], vocab_size]
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []

        for i in range(max_len):
            # returns torch.Size([1,1,hidden_size])
            lstm_out, states = self.lstm(inputs, states)

            # returns torch.Size([1,1,vocab_size])
            outputs = self.linear(lstm_out)

            # returns torch.Size([1,vocab_size])
            outputs = outputs.squeeze(1)
            word_id  = outputs.argmax(dim=1)
            caption.append(word_id.item())
            
            # returns torch.Size([1,1,embed_size])
            inputs = self.word_embeddings(word_id.unsqueeze(0))
          
        return caption