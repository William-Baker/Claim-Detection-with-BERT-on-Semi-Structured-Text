from tqdm import tqdm
import pandas as pd
from copy import copy

def get_lists_of_sentences(data, split_characters, tokenizer, MODE, min_length=10):
    """
    Takes a corpus of text and splits it into a list of lists of entailing sentences
    Where each sentence is a list of tokens, tokenized using tokenizer
    Sentences are joined together/discarded if the min length is not met

    Args:
        data (pd.Series): Pandas Series containing a list of strings
        split_characters (list): A list of split characters to split the sentences with, such as '.'
        tokenizer (object): tokenizer implementing the tokenize function
        MODE (str): The current mode "Twitter" or "UKP"
        min_length (int, optional): minimum allowable sentence length. Defaults to 10.

    Returns:
        list(list(list)): list of lists of entailing sentences where each sentence is a list of tokens
    """

    # Process the text into a list of sentences, using all the split characters provided
    split = []
    for split_character in split_characters:
        split += [x.split(split_character) for x in data if not pd.isna(x)] # split into sentences with the the split character
        
    split = [list(filter(lambda i: len(i) > 0, x)) for x in split] # remove any empty sentences

    split = [[tokenizer.tokenize(i) for i in x] for x in tqdm(split, desc='Tokenizing Text')]

    # Unfortunately the Twitter tokenizer doesnt recognise special tokens such as [TAG] so splits them into [, tag, ]. So we have to repair these changes
    if MODE=='Twitter':
        special = {
            str(['[','rt',']']):['[RT]'],
            str(['[','men',']']):['[MEN]'],
            str(['[','tag',']']):['[TAG]'],
            str(['[','tag',']']):['[TAG]'],
        }

        def repair_special_tokens(x):
            for index in range(0, len(x)-2):
                if str(x[index:index+3]) in special:
                    x = x[0:index] + special[str(x[index:index+3])] + x[index+3:]
                    return repair_special_tokens(x)
            return x

        split = [[repair_special_tokens(i) for i in x] for x in tqdm(split, desc='Repairing Text')]


    def lis_merger(lis):
        """
        Given a dataset of lists of sentences - where each sentence is a list of tokens,
        Merge sentences shorter than [min_length] together until the dataset only contains
        lists of sentences of length > [min_length]
        """
        length = len(lis)
        lis = copy(lis)
        for index in range(length):
            item = lis[index]

            restart = False # whether we need to re-call the function because the list was edited
            if len(item) < min_length: # We're too short
                if index > 0 and index + 1 < length: # there are sentences either side of us
                    prev = lis[index-1]
                    next = lis[index+1]
                    if len(prev) < len(next):  # The prior sentence is shorter, merge
                        lis[index-1] = prev + item
                        restart = True
                    else:                      # The following sentence is shorter, merge
                        lis[index+1] = item + next
                        restart = True
                    lis.pop(index)
                elif index > 0: # We're the final sentence in the list, merge with the prior
                    lis[index-1] = lis[index-1] + item
                    restart = True
                    lis.pop(index)
                elif index + 1 < length: # We're the first sentence in the list, merge with whatever follows
                    lis[index+1] = item + lis[index+1]
                    restart = True
                    lis.pop(index)
                else: # Give up
                    return []
            
            if restart:
                return lis_merger(lis)
        return lis
        
    split = [lis_merger(x) for x in tqdm(split, desc='Merging Sentences')]

    return split

import torch

BERT_MODEL_NAME = "bert-base-uncased" # bert-base-uncased bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese
from pytorch_pretrained_bert import BertModel

class BERTClassification(torch.nn.Module):
    """
    The BERT model with a classification layer using SoftMax appended
    """
    def __init__ (self, output_size):
        super(BERTClassification, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.bert_drop = torch.nn.Dropout(0.1)
        self.out_dense = torch.nn.Linear(768, output_size)
        self.activation = torch.nn.Softmax(dim=1)
        
    def forward(self, ids, mask, token_type_ids):
        bert_hidden, pooledOut = self.bert(ids, attention_mask = mask,
                                token_type_ids=token_type_ids)
        bertOut = self.bert_drop(pooledOut)
        z = self.out_dense(bertOut)
        out = self.activation(z)
        
        return out, bertOut, pooledOut, bert_hidden