import torch
import torch.nn as nn

class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Dropout randomly "turns off" neurons during training 
        # to force the AI to learn broader patterns.
        self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        # Apply dropout to the word vectors
        dropped = self.dropout(embedded)
        pooled = dropped.mean(dim=1) 
        return self.fc(pooled)