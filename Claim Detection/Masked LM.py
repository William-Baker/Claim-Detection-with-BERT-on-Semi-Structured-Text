#%%

# Import all the dependencies
import pandas as pd
import re
from copy import copy 
import random
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from transformers import BertTokenizer
from common import get_lists_of_sentences
import torch
from common import BERTClassification
from pytorch_pretrained_bert import BertAdam

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, IterableDataset




# Set the seed for all random number generators, this makes the results more reproducible
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)






# Select the MODE to run in, these correspond to the 2 Masked LM tasks on the Twitter, and UKP datasets
# Simply paste the string in from the options bellow to change the task
# 'Twitter'
# 'UKP'
MODE = 'Twitter'

# Here the Hyper parameters for the model are chosen
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5

# The bert model loaded that will be trained to perform the Masked LM task
BERT_MODEL = "bert-base-uncased" # bert-base-uncased bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese










#%%


tokenizer = None

if MODE=='Twitter':
    # Import the Twitter Dataset (Un-Annotated)
    df = pd.read_pickle('../datasets/cleaned tweets.pkl')
    df.rename(columns={'text':'x'}, inplace=True)
    train_raw = df # Since we aren't analysing the performance of this model, just train on all the data
    # Load the tokenizer with the Twitter vocabulary
    tokenizer = BertTokenizer.from_pretrained('../bert-it/bert-it-vocab.txt')
    log_dir = 'Trained Models/Twitter/Masked LM/' # Write the logs and models to the appropriate folder

else:
    # Import the UKP dataset
    df = pd.read_excel('../datasets/UKP Claim Detection.xlsx')
    # The UKP datasett has a couple URL's in, remove them
    url_regex = re.compile("""(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})""")
    df['x'] = df['x'].apply(lambda x: url_regex.sub("", x) if not pd.isna(x) else x)
    
    train_raw = df # Since we aren't analysing the performance of this model, just train on all the data

    # Load the tokenizer with the default vocabulary
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    log_dir = 'Trained Models/UKP/Masked LM/' # Write the logs and models to the appropriate folder


summary_writer = SummaryWriter(log_dir + 'runs') # Create the tensorboard logger

# This is where we check if the system has a GPU, if so use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




#%%




# Convert the tokens to ID's and compute the appropriate segment ids and input masks that bert expects
def preprocess(sentences):
    """
    Converts each sentence of tokens into the corresponding input id's, segment id's and input masks
    Truncating each sequence to the MAX_SEQ_LENGTH
    """
    df = pd.DataFrame()
    df['tokens'] = [["[CLS]"] + x + ["[SEP]"] for x in sentences]
    df['segment ids'] = [[0] * MAX_SEQ_LENGTH] * len(sentences)
    df['x'] = df['tokens'].apply(tokenizer.convert_tokens_to_ids)
    df['x'] = df['x'].apply(lambda x: x[0:MAX_SEQ_LENGTH-1] + tokenizer.convert_tokens_to_ids(["[SEP]"]) if len(x) > MAX_SEQ_LENGTH else x)
    df['length'] = df['x'].apply(len)
    df['padding'] = df['x'].apply(lambda x: [0] * (MAX_SEQ_LENGTH - len(x)))
    df['input mask'] = df['x'].apply(lambda x: [1] * len(x)) + df['padding']
    df['x'] = df['x'] + df['padding']
    return  df






#%%

# Only test in the Twitter mode as that tests the full functionality of the get_lists_of_sentences function
if MODE == 'Twitter':
    def test_preprocess():
        """
        Tests the functionality of get_lists_of_sentences and preprocess
        Assertions will throw exceptions if an error occurs
        """
        # Test the tokenization
        test_input = "[RT] [MEN] hello there. the quick brown fox jumped over the hill [TAG]"
        target_1 = [tokenizer.convert_tokens_to_ids(x) for x in ["[CLS]", "[RT]", "[MEN]", "hello", "there", "[SEP]"]]
        target_2 = [tokenizer.convert_tokens_to_ids(x) for x in ["[CLS]", "the", "quick", "brown", "fox", "jumped", "over", "the", "hill", "[TAG]", "[SEP]"]]
        test_sentences = get_lists_of_sentences([test_input], ['.'], tokenizer, MODE, min_length=1)  
        test_sentences = sum(test_sentences, [])
        processed_test_sentences = preprocess(test_sentences)
        assert processed_test_sentences.iloc[0]['x'][:len(target_1)] == target_1
        assert processed_test_sentences.iloc[1]['x'][:len(target_2)] == target_2

        # Test the sentence merging is working
        test_sentences = get_lists_of_sentences([test_input], ['.'], tokenizer, MODE, min_length=10)  
        test_sentences = sum(test_sentences, [])
        processed_test_sentences = preprocess(test_sentences)
        target_3 = target_1[:-1] + target_2[1:]
        assert processed_test_sentences.iloc[0]['x'][:len(target_3)] == target_3

    test_preprocess()





#%%

# Split each tweet into sentences and tokenize them, then merge sentences that are too short (minimum length of 10 by default)
sentences = []
if MODE=='Twitter':
    sentences = get_lists_of_sentences(train_raw['x'], ['.'], tokenizer, MODE)
else:
    sentences = get_lists_of_sentences(train_raw['x'], ['.', ';', ','], tokenizer, MODE) # Ideally we'd only use '.' but thanks to the small dataset use more


# Merge the list of lists of senteces into a single list of sentences
# Ideally we would use sum (shown bellow), but it does not always terminate for large inputs, instead we compute the sum ourselves
# sentences = sum(sentences, [])
acc = []
for x in sentences:
    acc += x
sentences = acc

train_x = preprocess(sentences)





#%%





mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
vocab_size = len(tokenizer.vocab)


def generate_sample(sample):
    """
    Construct the generator that can convert a simple sentence into a sentence with 15% of the words masked off, according to Googles paper
    80% of those words we mask, replace with the [MASK] token
    10% with a random token
    10% unchanged
    In all cases add the masks word id's to the target vector
    """
    sample_length = sample['length']
    samples_to_mask = int(0.15 * sample_length)
    indices = random.sample(range(1, sample_length-1), samples_to_mask) # take 1 off each end to account for the [CLS] and [SEP] tags

    masked_input = copy(sample['x'])

    # Targets are all the id's of the words we're masking
    y = [0] * vocab_size
    for index in indices:
        token = masked_input[index]
        y[token] = 1

    # Mask of the indices we've chosen
    for index in indices:
        rand = random.random()
        if rand < 0.8:
            masked_input[index] = mask_id # replace token with [MASK] token
        elif rand < 0.9:
            masked_input[index] = random.randint(0, vocab_size-1) # replace with a random token
        # otherwise leave index unchanged
    
    return (torch.tensor(masked_input), 
            torch.tensor(sample['input mask']), 
            torch.tensor(sample['segment ids']),
            torch.tensor(y, dtype=torch.float))








#%%

# Create the data generator, rather than generate all the possible samples, saving lots of memory
# Also handles randomly sampling from the dataset
class MaskedLMData(IterableDataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
    def __iter__(self):
        return iter(generate_sample(x) for _, x in self.df.sample(frac=1).iterrows())

# This constructs batches of samples using the data generator, and pre-emptily sends them to the device
train_dataloader = DataLoader(MaskedLMData(train_x), batch_size=BATCH_SIZE,  
    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

num_train_steps = np.ceil(train_x.shape[0] / BATCH_SIZE)






#%%

# Load the default BERT model, and add a classification layer with the same number of outputs as the size of out vocabulary
model = BERTClassification(vocab_size)

model.to(device) # Move the model to the compute device

# Many papers seem to modify the weight decay to prevent bias and layerNorms being decayed
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=LEARNING_RATE,
                            warmup=0.1,
                            t_total=num_train_steps*EPOCHS)



global_step = 0


#%%





loss_function = torch.nn.BCELoss() # Use the BCE loss as we're doing multi-label classification

def epoch():
    global global_step
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", total=num_train_steps)):
        input_ids, input_mask, segment_ids, label_ids = batch # unpack the tuple from the data loader
        out, bertOut, pooledOut, bert_hidden = model(input_ids, segment_ids, input_mask) # make our predictions for this batch
        loss = loss_function(out, label_ids) # measure how good the predictions were
        loss.backward() # backpropigate the error, updating the model
                        
        #  log the loss to tensorboard
        summary_writer.add_scalar('Loss/train', loss, global_step)

        # iterate the optimizer and reset it's gradients
        optimizer.step()
        optimizer.zero_grad()
        
        global_step += 1

    torch.save(model, log_dir + f"{pd.Timestamp.now()} step: {global_step}") # save the model from this epoch


#%%






def eval(eval_data):
    global global_step

    sentences = get_lists_of_sentences(eval_data.iloc[0:256], ['.', ';', ','], tokenizer, MODE)
    sentences = sum(sentences, [])

    test_x = preprocess(sentences)
    test_dataloader = DataLoader(MaskedLMData(test_x), batch_size=BATCH_SIZE,  
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    model.eval()
    accuracy = []

    accuracy_list = []
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):

        with torch.no_grad():
            out, bertOut, pooledOut, bert_hidden = model(input_ids, segment_ids, input_mask)

        # Transfer the predictions and targets to the CPU
        pred = out.detach().cpu().numpy()
        targ = label_ids.to('cpu').numpy()
        
        acc = []
        for batch_index in range(0, pred.shape[0]):
            p = pred[batch_index,:]
            t = targ[batch_index,:]
            target_count = int(t.sum()) # the sum of the target vector will equal the number tokens we masked
            target_indices = (-t).argsort()[:target_count] # calculate the top predicted targets
            predicted_indices = (-p).argsort()[:target_count]

            accuracy = len(set.intersection(set(target_indices), set(predicted_indices))) / target_count

            predicted_ones = np.zeros(p.shape)
            predicted_ones[predicted_indices] = 1

            acc.append(accuracy)

    
        accuracy_list.append(mean(acc))

    # Only calculate accuracy, since there are 30k classes F1 is expensive
    accuracy = mean(accuracy_list)
    print(f"Accuracy: {accuracy}%")
    summary_writer.add_scalar('Accuracy', accuracy, global_step)




# %%

for i in range(EPOCHS):
    epoch()
    eval(df['x'])

#%%