#%%
#%%
import pandas as pd
import re
from copy import copy 
import random
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
from torch.utils.data.dataloader import default_collate
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch
from common import get_lists_of_sentences
from common import BERTClassification
from pytorch_pretrained_bert import BertAdam


# Set the seed for all random number generators, this makes the results more reproducible
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)






# Select the MODE to run in, these correspond to the 2 NSP tasks on the Twitter, and UKP datasets
# Simply paste the string in from the options bellow to change the task
# 'Twitter'
# 'UKP'
MODE = 'Twitter'

# Here the Hyper parameters for the model are chosen
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5

# The bert model loaded that will be trained to perform the MSP task
BERT_MODEL = "bert-base-uncased" # bert-base-uncased bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese

# Specify the path to the pre-trained Masked LM model
if MODE=='Twitter':
    MASKED_LM_PATH = "Trained Models/Twitter/Masked LM/2022-04-01 06:10:01.498390 step: 88655"
else:
    MASKED_LM_PATH = "Trained Models/UKP/Masked LM/2022-04-01 00:39:32.682074 step: 25220"










#%%


tokenizer = None

if MODE=='Twitter':
    # Import the Twitter Dataset (Un-Annotated)
    df = pd.read_pickle('../datasets/cleaned tweets.pkl')
    df.rename(columns={'text':'x'}, inplace=True)
    train_raw = df # Since we aren't analysing the performance of this model, just train on all the data
    # Load the tokenizer with the Twitter vocabulary
    tokenizer = BertTokenizer.from_pretrained('../bert-it/bert-it-vocab.txt')
    log_dir = 'Trained Models/Twitter/NSP/' # Write the logs and models to the appropriate folder

else:
    # Import the UKP dataset
    df = pd.read_excel('../datasets/UKP Claim Detection.xlsx')
    # The UKP datasett has a couple URL's in, remove them
    url_regex = re.compile("""(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})""")
    df['x'] = df['x'].apply(lambda x: url_regex.sub("", x) if not pd.isna(x) else x)
    
    train_raw = df # Since we aren't analysing the performance of this model, just train on all the data

    # Load the tokenizer with the default vocabulary
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    log_dir = 'Trained Models/UKP/NSP/' # Write the logs and models to the appropriate folder


summary_writer = SummaryWriter(log_dir + 'runs') # Create the tensorboard logger

# This is where we check if the system has a GPU, if so use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






#%%

# Convert the tokens to ID's and compute the appropriate segment ids and input masks that bert expects
def preprocess(labelled_sentence_pairs):
    """
    Converts each pair of sentences of tokens into the corresponding input id's, segment id's and input masks
    Truncating each sequence to the MAX_SEQ_LENGTH
    """
    df = pd.DataFrame()
    df['tokens'] = labelled_sentence_pairs.apply(lambda e: ["[CLS]"] + e['A'] + ["[SEP]"] + e['B'] + ["[SEP]"], axis=1)
    df['segment ids'] = labelled_sentence_pairs.apply(lambda e: [0] * len(["[CLS]"] + e['A'] + ['[SEP]'])
                                                                            + [1] * len(e['B'] + ["[SEP]"])  , axis=1)
    df['x'] = df['tokens'].apply(tokenizer.convert_tokens_to_ids)
    df['padding'] = df['x'].apply(lambda x: [0] * (MAX_SEQ_LENGTH - len(x)))
    df['input mask'] = df['x'].apply(lambda x: [1] * len(x)) + df['padding']
    df['x'] = df['x'] + df['padding']
    df['segment ids'] = df['segment ids'] + df['padding']
    df['y'] = labelled_sentence_pairs['y']
    return  df

def generate_sentence_pairs(sentences):
    sentences = [x for x in sentences if len(x) > 1] # only keep lists of sentences with 2+ sentences
    
    #valid_pairs = sum([[x[i:i+2] for i in range(0, len(x)-1)] for x in sentences], [])
    valid_pairs = []
    for x in tqdm(sentences, desc="listing pairs of sentences"):
        for i in range(0, len(x) - 1):
            valid_pairs.append(x[i:i+2])

    # Pick rabdom pairs, provided the pair doesnt originate from the same original text
    invalid_pairs = []
    for invalid_sample_count in tqdm(range(len(valid_pairs)), desc="generating invalid pairs of sentences"):
        sentence_list_a_index = random.randint(0, len(sentences)-1)
        
        # Pick a random index not equal to our other source
        sentence_list_b_index = random.randint(0, len(sentences)-2)
        if sentence_list_b_index >= sentence_list_a_index:
            sentence_list_b_index += 1

        # select random sentences from each sentece list
        a = random.choice(sentences[sentence_list_a_index])
        b = random.choice(sentences[sentence_list_b_index])

        invalid_pairs.append([a,b])

    valid_df = pd.DataFrame(valid_pairs, columns=['A', 'B'])
    valid_df['y'] = 1

    invalid_df = pd.DataFrame(invalid_pairs, columns=['A', 'B'])
    invalid_df['y'] = 0

    return pd.concat([valid_df, invalid_df]).sample(frac=1)




#%%



# Only test in the Twitter mode as that tests the full functionality of the get_lists_of_sentences function
if MODE == 'Twitter':
    def test_preprocess():
        """
        Tests the functionality of get_lists_of_sentences and generate_sentence_pairs and preprocess methods
        Assertions will throw exceptions if an error occurs
        """
        # Test the tokenization
        test_input_1 = "[RT] [MEN] hello there. the quick brown fox jumped over the hill [TAG]"
        test_input_2 = "[MEN] [MEN] a fat cat on a mat. on a mat that cat had sat"
        target_1 = [tokenizer.convert_tokens_to_ids(x) for x in ["[CLS]", "[RT]", "[MEN]", "hello", "there", "[SEP]", "the", "quick", "brown", "fox", "jumped", "over", "the", "hill", "[TAG]", "[SEP]"]]
        target_2 = [tokenizer.convert_tokens_to_ids(x) for x in ["[CLS]", "[MEN]", "[MEN]", "a", "fat", "cat", "on", "a", "mat", "[SEP]", "on", "a", "mat", "that", "cat", "had", "sat", "[SEP]"]]
        test_sentences = get_lists_of_sentences([test_input_1, test_input_2], ['.'], tokenizer, MODE, min_length=1)
        test_sentences_pairs = generate_sentence_pairs(test_sentences)
        processed_test_sentences = preprocess(test_sentences_pairs)
        true_processed_test_sentences = processed_test_sentences[processed_test_sentences['y'] == 1]
        
        assert true_processed_test_sentences.iloc[0]['x'][:len(target_1)] == target_1 or true_processed_test_sentences.iloc[0]['x'][:len(target_2)] == target_2
        assert true_processed_test_sentences.iloc[1]['x'][:len(target_2)] == target_2 or true_processed_test_sentences.iloc[1]['x'][:len(target_1)] == target_1
        

    test_preprocess()
else:
    def test_preprocess():
        """
        Tests the functionality of get_lists_of_sentences and generate_sentence_pairs and preprocess methods
        Assertions will throw exceptions if an error occurs
        """
        # Test the tokenization
        test_input_1 = "hello there. the quick brown fox jumped over the hill"
        test_input_2 = "a fat cat on a mat. on a mat that cat had sat"
        target_1 = [tokenizer.convert_tokens_to_ids(x) for x in ["[CLS]", "hello", "there", "[SEP]", "the", "quick", "brown", "fox", "jumped", "over", "the", "hill", "[SEP]"]]
        target_2 = [tokenizer.convert_tokens_to_ids(x) for x in ["[CLS]", "a", "fat", "cat", "on", "a", "mat", "[SEP]", "on", "a", "mat", "that", "cat", "had", "sat", "[SEP]"]]
        test_sentences = get_lists_of_sentences([test_input_1, test_input_2], ['.'], tokenizer, MODE, min_length=1)
        test_sentences_pairs = generate_sentence_pairs(test_sentences)
        processed_test_sentences = preprocess(test_sentences_pairs)
        true_processed_test_sentences = processed_test_sentences[processed_test_sentences['y'] == 1]
        
        assert true_processed_test_sentences.iloc[0]['x'][:len(target_1)] == target_1 or true_processed_test_sentences.iloc[0]['x'][:len(target_2)] == target_2
        assert true_processed_test_sentences.iloc[1]['x'][:len(target_2)] == target_2 or true_processed_test_sentences.iloc[1]['x'][:len(target_1)] == target_1
        

    test_preprocess()




#%%




# Split each tweet into sentences and tokenize them, then merge sentences that are too short (minimum length of 10 by default)
sentences = []
if MODE=='Twitter':
    sentences = get_lists_of_sentences(train_raw['x'], ['.'], tokenizer, MODE)
else:
    sentences = get_lists_of_sentences(train_raw['x'], ['.', ';', ','], tokenizer, MODE) # Ideally we'd only use '.' but thanks to the small dataset use more


# given the list of lists of sentences, convert these into valid and invalid pairs
labelled_sentence_pairs = generate_sentence_pairs(sentences)



# Before converting the sentence into IDs, make sure that together they are 3 tokens less than the MAX_SEQ_LENGTH
def shorten_sentence_pair(a, b, target):
    """
    Shotens a pair of lists by removing 1 element the longest list repeatedly until the desired length is achieved
    """
    while len(a) + len(b) > target:
        if len(a) > len(b):
            a = a[:-1]
        else:
            b = b[:-1]
    return (a,b)
# Apply the shortening function to the whole dataset
labelled_sentence_pairs[['A', 'B']] = labelled_sentence_pairs.apply(lambda e: shorten_sentence_pair(e['A'], e['B'], MAX_SEQ_LENGTH-3), axis=1, result_type="expand")

    
# Generate the input ids, input masks and segment ids
train = preprocess(labelled_sentence_pairs)



#%%




# Convert the dataset into tensors, for Pytorch compatability
tensor_dataset = TensorDataset(torch.tensor(train['x'].tolist()),
                                torch.tensor(train['input mask'].tolist()),
                                torch.tensor(train['segment ids'].tolist()),
                                torch.tensor(train['y'].tolist()))

# Use a random sampler to uniformly sample training examples
train_random_sampler = RandomSampler(tensor_dataset)

# This constructs batches of samples, and pre-emptily sends them to the device
train_dataloader = DataLoader(tensor_dataset, sampler=train_random_sampler, batch_size=BATCH_SIZE,  
    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

num_train_steps = int(np.ceil(train.shape[0] / BATCH_SIZE))




#%%









# Load the default BERT model, and add a classification layer with 2 outputs, either sentence 
model = BERTClassification(2)

model_masked_lm = torch.load(MASKED_LM_PATH) # Load the Masked LM Model we just trained

model.bert = model_masked_lm.bert # Only replace the BERT embeddings and self-attention layers, we're adding a new classification layer


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




loss_function = torch.nn.CrossEntropyLoss() # automatically recognises sparse labels unlike TF's implementation

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






import numpy as np
from statistics import mean
from sklearn.metrics import f1_score
def eval(eval_data):
    global global_step
    
    sentences = []
    if MODE=='Twitter':
        sentences = get_lists_of_sentences(train_raw['x'], ['.'], tokenizer, MODE)
    else:
        sentences = get_lists_of_sentences(train_raw['x'], ['.', ';', ','], tokenizer, MODE) # Ideally we'd only use '.' but thanks to the small dataset use more
    labelled_sentence_pairs = generate_sentence_pairs(sentences)
    labelled_sentence_pairs[['A', 'B']] = labelled_sentence_pairs.apply(lambda e: shorten_sentence_pair(e['A'], e['B'], MAX_SEQ_LENGTH-3), axis=1, result_type="expand")
    test = preprocess(labelled_sentence_pairs)
    tensor_dataset = TensorDataset(torch.tensor(test['x'].tolist()),
                                torch.tensor(test['input mask'].tolist()),
                                torch.tensor(test['segment ids'].tolist()),
                                torch.tensor(test['y'].tolist()))

    test_random_sampler = RandomSampler(tensor_dataset)

    test_dataloader = DataLoader(tensor_dataset, sampler=test_random_sampler, batch_size=BATCH_SIZE,  
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))  



    model.eval()
    accuracy = []

    p = []
    t = []
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):

        with torch.no_grad():
            out, bertOut, pooledOut, bert_hidden = model(input_ids, segment_ids, input_mask)
        
        # Transfer the predictions and targets to the CPU
        pred = out.detach().cpu().numpy()
        targ = label_ids.to('cpu').numpy()

        pred_s = np.argmax(pred, axis=1).flatten()
        accuracy.append( (pred_s==targ.flatten()).mean() )

        p += list(pred_s)
        t += list(targ.flatten())

    print(f"Accuracy: {mean(accuracy)}%")
    summary_writer.add_scalar('Accuracy', mean(accuracy), global_step)

    f1 = f1_score(t, p, average="micro")
    print(f"F1:       {f1}")
    summary_writer.add_scalar('F1 Score', f1, global_step)

    test['prediction'] = p
    test.to_excel('pred.xlsx')




#%%

for i in range(0, EPOCHS):
    epoch()
    eval(df['x'].sample(256))

# %%

