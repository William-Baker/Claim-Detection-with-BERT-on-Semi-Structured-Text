#%%

# Import all the dependencies
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertAdam
from sklearn.preprocessing import OrdinalEncoder
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import tqdm
import numpy as np
from statistics import mean
from sklearn.metrics import f1_score
from common import BERTClassification




# Select the MODE to run in, these correspond to the 4 claim detection tasks: UT, TT, TU and UU
# Simply paste the string in from the options bellow to change the task
# TT : 'Twitter Pretrained'
# UT : 'Twitter Default'
# TU : 'UKP Pretrained'
# UU : 'UKP Default'
MODE = 'Twitter Pretrained'

# Here the Hyper parameters for the model are chosen
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 4
BINARIZE_LABELS = True # Whethere to binarise the UKP dataset from (For,Against,None) to (Claim/No Claim)

# The bert model loaded in the Default case - for a fair comparision the BERT MODEL used in each task should be the same
BERT_MODEL = "bert-base-uncased" # bert-base-uncased bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese


# Configure which pre-trained model to load, if any
if MODE=='UKP Default': 
    pre_trained_model_path = None
elif MODE=='UKP Pretrained':
    pre_trained_model_path = 'Trained Models/UKP/NSP/2022-03-12 16:38:17.119496 step: 20850'
elif MODE=='Twitter Default':
    pre_trained_model_path = None
elif MODE=='Twitter Pretrained':
    pre_trained_model_path = 'Trained Models/Twitter/NSP/2022-04-01 18:58:41.246851 step: 46515'



#%%

if MODE=='UKP Default': # Write the logs and models to the appropriate folder
    log_dir = 'Trained Models/UKP/Claim Detection Default/'
elif MODE=='UKP Pretrained':
    log_dir='Trained Models/UKP/Claim Detection Pretrained/'
elif MODE=='Twitter Default':
    log_dir='Trained Models/Twitter/Claim Detection Default/'
elif MODE=='Twitter Pretrained':
    log_dir='Trained Models/Twitter/Claim Detection Pretrained/'


summary_writer = SummaryWriter(log_dir + 'runs') # Create the tensorboard logger

# This is where we check if the system has a GPU, if so use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the seed for all random number generators, this makes the results more reproducible
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)




# Load the appropriate dataset depending on the mode
if 'Twitter' in MODE:
    df = pd.read_pickle('../datasets/labelled tweets.pkl')
else:
    df = pd.read_excel('../datasets/UKP Claim Detection.xlsx')
    if BINARIZE_LABELS:
        df['y'] = df['y'].apply(lambda x: 'NoArgument' if x=='NoArgument' else 'Argument' )


if MODE == 'Twitter Pretrained': # Only use the Twitter specific vocabulary if we've pre-trained on the Twitter data
    tokenizer = BertTokenizer.from_pretrained('../bert-it/bert-it-vocab.txt')
else:
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

df.dropna(subset=['x'], inplace=True)

# Split the dataset into the various train/test/validation splits
train_raw = pd.DataFrame(df[df['split'] == 'train'])
test_raw = pd.DataFrame(df[df['split'] == 'test'])
val_raw = pd.DataFrame(df[df['split'] == 'val'])








#%%

# try and load the current ordinal encoder, this ensures that we're using the correct target mappings, if you wish to skip training and only evaluate the model
try:
    preprocessor_config = pd.read_pickle('preprocessor.pkl')
    targ_encoder = preprocessor_config.loc['targ_encoder']['x']
except:
    targ_encoder = OrdinalEncoder()
    preprocessor_config = pd.DataFrame()


#%%

def preprocess_data(df, fit=False):
    """
    Preprocesses the raw text data to produce Input IDs, Input Masks and Segment IDs

    Args:
        df (DataFrame): A dataframe containing a text column 'x' with the input data, and 'y' with target class labels (also a 'topic' column for UKP tasks)
        fit (bool, optional): Whether to fit the target encoder (False if evaluating, True if training a new model) Defaults to False.

    Returns:
        DataFrame: df with the Input IDs ('x'), Input Masks ('input mask')  and Segment IDs ('segment ids') columns appended
    """
    processed = pd.DataFrame()
    processed['x tokens'] = df['x'].apply(tokenizer.tokenize)

     # Unfortunately the BERT tokenizer doesn't recognise special tokens such as [TAG] so splits them into [, tag, ]. So we have to repair these changes
    if 'Twitter' in MODE:
        special = {
            str(['[','rt',']']): ['[RT]'],
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

        processed['x tokens'] = processed['x tokens'].apply(repair_special_tokens)

    

    # If we are fitting the data, fit a new ordinal encoder and save it to a file
    if fit:
        targets_numpy = df['y'].to_numpy().reshape(-1, 1)
        processed['y'] = list(targ_encoder.fit_transform(targets_numpy).flatten())
        preprocessor_config = pd.DataFrame( pd.Series({'x': targ_encoder}, name='targ_encoder'))
        preprocessor_config.to_pickle('preprocessor.pkl')
    else:
        targets_numpy = df['y'].to_numpy().reshape(-1, 1)
        processed['y'] = list(targ_encoder.transform(targets_numpy).flatten())
    processed['y'] = processed['y'].astype(np.int32)

    if "UKP" in MODE: # UKP data contains topic information, so must be split into two sentences (and each shortened to MAX_SEQ_LENGTH)
        def shorten(entity, fields):
            """Shortens a number of fields, such that their sum is less than MAX_SEQ_LENGTH-3

            Args:
                entity (pd.Series): A series containing the set of fields
                fields (list): list of fields to shorten

            Returns:
                pd.Series: the shortened entity
            """
            quota = MAX_SEQ_LENGTH-3
            new_entity = pd.Series(entity)
            
            def get_lengths():
                len_dict = dict()
                for field in fields:
                    len_dict[field] = len(new_entity[field])
                return len_dict
            len_dict = get_lengths()
            
            while sum(len_dict.values()) > quota:
                worst_field = max(len_dict, key=len_dict.get) 
                new_entity[worst_field] = new_entity[worst_field][:-1]
                len_dict = get_lengths()

            return new_entity

        
        processed['topic tokens'] = df['topic'].apply(tokenizer.tokenize)
        processed = processed.apply(lambda e: shorten(e, ['topic tokens', 'x tokens']), axis=1)
        processed['tokens'] = processed.apply(lambda e: ["[CLS]"] + e['topic tokens'] + ["[SEP]"] + e['x tokens'] + ["[SEP]"], axis=1)
        processed['segment ids'] = processed.apply(lambda e: [0] * len(["[CLS]"] + e['topic tokens'] + ['[SEP]'])
                                                    + [1] * len(e['x tokens'] + ["[SEP]"])  , axis=1)
    else: # Twitter data contains topic information, so must be split into two sentences (and each shortened to MAX_SEQ_LENGTH)
        def shorten(x):
            """Simply shortens the input sequence to be 3 tokens shorter than MAX_SEQ_LENGTH
            """
            if len(x) > MAX_SEQ_LENGTH-3:
                return x[:MAX_SEQ_LENGTH-3]
            else:
                return x

        processed['x tokens'] = processed['x tokens'].apply(shorten)
        processed['tokens'] = processed['x tokens'].apply(lambda x: ["[CLS]"] + x + ["[SEP]"])
        processed['segment ids'] = processed.apply(lambda e: [0] * len(["[CLS]"] + e['x tokens'] + ['[SEP]'])  , axis=1)
    
    # Compute the input Ids, input masks and segment ids
    processed['x'] = processed['tokens'].apply(tokenizer.convert_tokens_to_ids)
    processed['padding'] = processed['x'].apply(lambda x: [0] * (MAX_SEQ_LENGTH - len(x)))
    processed['input mask'] = processed['x'].apply(lambda x: [1] * len(x)) + processed['padding']
    processed['x'] = processed['x'] + processed['padding']
    processed['segment ids'] = processed['segment ids'] + processed['padding']

    # Validate the lengths of the generated sequences
    assert processed['x'].apply(len).min()           == processed['x'].apply(len).max() == MAX_SEQ_LENGTH
    assert processed['input mask'].apply(len).min()  == processed['input mask'].apply(len).max() == MAX_SEQ_LENGTH
    assert processed['segment ids'].apply(len).min() == processed['segment ids'].apply(len).max() == MAX_SEQ_LENGTH

    return processed



def test_preprocess():
    """
    Tests the functionality of preprocess data, according to the current MODE
    Assertions will throw exceptions if an error occurs
    """
    test_input_a = "that's good!"
    test_input_b = "no good"
    tokenized_test_input_a = tokenizer.tokenize(test_input_a)
    tokenized_test_input_b = tokenizer.tokenize(test_input_b)
    assert tokenized_test_input_a == ['that', "'", 's', 'good', '!']
    assert tokenized_test_input_b == ['no', 'good']

    #                        Tests       A             B
    test_data = pd.DataFrame(data=[
                                    [test_input_a,'good',1],
                                    [test_input_b, ' '.join(['bad']*MAX_SEQ_LENGTH),0]], # have 1 very long topic

                    columns=['x', 'topic', 'y'])


    proc = preprocess_data(test_data, fit=True)
    res_a = proc.iloc[0]
    res_b = proc.iloc[1]
    assert (res_a['y'] == 0 and  res_b['y'] == 1) or (res_a['y'] == 1 and  res_b['y'] == 0)
    if 'Twitter' in MODE:
        a_len = len(tokenized_test_input_a)+2
        assert res_a['x'][0:a_len] == tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_test_input_a + ['[SEP]'])
        assert res_a['input mask'] == [1] * a_len + [0] * (MAX_SEQ_LENGTH - a_len)
        assert res_a['segment ids'] == [0] * MAX_SEQ_LENGTH

        b_len = len(tokenized_test_input_b)+2
        assert res_b['x'][0:b_len] == tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_test_input_b + ['[SEP]'])
        assert res_b['input mask'] == [1] * b_len + [0] * (MAX_SEQ_LENGTH - b_len)
        assert res_b['segment ids'] == [0] * MAX_SEQ_LENGTH
    else:
        a_len = len(tokenized_test_input_a)+4
        assert res_a['x'][0:a_len] == tokenizer.convert_tokens_to_ids(['[CLS]'] + ['good'] + ['[SEP]'] + tokenized_test_input_a + ['[SEP]'])
        assert res_a['input mask'] == [1] * a_len + [0] * (MAX_SEQ_LENGTH - a_len)
        assert res_a['segment ids'] == [0] * 3 + [1]*(len(tokenized_test_input_a)+1) + [0] * (MAX_SEQ_LENGTH-3-len(tokenized_test_input_a)-1)

        b_len_minus_topic = len(tokenized_test_input_b) + 3
        b_len_topic = MAX_SEQ_LENGTH - b_len_minus_topic
        assert res_b['x'] == tokenizer.convert_tokens_to_ids(['[CLS]'] + ['bad'] * b_len_topic + ['[SEP]'] + tokenized_test_input_b + ['[SEP]'])
        assert res_b['input mask'] == [1] * MAX_SEQ_LENGTH
        assert res_b['segment ids'] == [0] * (b_len_topic  + 2)  + [1] * (b_len_minus_topic - 2)
    
    if 'Twitter' in MODE:
        test_input_c = "[RT] [MEN] hello there [TAG]"
        target = [tokenizer.convert_tokens_to_ids(x) for x in ["[CLS]", "[RT]", "[MEN]", "hello", "there", "[TAG]", "[SEP]"]]
        test_data = pd.DataFrame(data=[ [test_input_c,'good',1]], columns=['x', 'topic', 'y'])
        res = preprocess_data(test_data, fit=True).iloc[0]
        assert res['x'][:len(target)] == target
        assert res['input mask'] == [1] * len(target) + [0] * (MAX_SEQ_LENGTH - len(target))
        assert res['segment ids'] == [0] * MAX_SEQ_LENGTH

    

test_preprocess()




#%%

train = preprocess_data(train_raw, fit=True)

#%%

# Convert the dataset into tensors, for Pytorch compatability
def to_dataset(df):
    return TensorDataset(
        torch.tensor(df['x'].tolist(), dtype=torch.long),
        torch.tensor(df['input mask'].tolist(), dtype=torch.long),
        torch.tensor(df['segment ids'].tolist(), dtype=torch.long),
        torch.tensor(df['y'].tolist(), dtype=torch.long)
    )
train_data = to_dataset(train)

# Use a random sampler to uniformly sample training examples
train_sampler = RandomSampler(train_data)

# This constructs batches of samples, and pre-emptily sends them to the device
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE,  
    collate_fn=lambda x: tuple(x_.to(device) for x_ in torch.utils.data.dataloader.default_collate(x)))

num_train_steps = int(np.ceil(len(train_data) / BATCH_SIZE))

#%%

# Load the default BERT model, and add a classification layer with 2 outputs (maybe 3 if training on the UKP corpus without binarisation)
model = BERTClassification(df['y'].unique().shape[0])

if pre_trained_model_path is not None:
    pre_trained = torch.load(pre_trained_model_path)
    model.bert = pre_trained.bert # replace the default BERT implementation with the pre-trained one, leaving our classification layer alone

model.to(device) # Move the model to the compute device

# Many papers seem to modify the weight decay to prevent bias and layerNorms being decayed
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Define the BERTAdam optimiser, this optimises the hyperparameters such as the learning rate during training
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
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, input_mask, segment_ids, label_ids = batch # unpack the tuple from the data loader
        out, bertOut, pooledOut, bert_hidden = model(input_ids, segment_ids, input_mask) # make our predictions for this batch
        loss = loss_function(out, label_ids) # measure how good the predictions were
        loss.backward() # backpropigate the error, updating the model
                        
        # Log the loss
        summary_writer.add_scalar('Loss/train', loss, global_step)

        # iterate the optimizer and reset it's gradients
        optimizer.step()
        optimizer.zero_grad()
        
        global_step += 1 

    torch.save(model, log_dir + f"{pd.Timestamp.now()} step: {global_step}") # save the model from this epoch

#%%



def eval(eval_data):
    global global_step
    test = preprocess_data(eval_data)
    test_data = to_dataset(test)
    test_sampler = SequentialSampler(test_data) # Sequentially sample the data, no need to randomise when evaluating
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE,  
            collate_fn=lambda x: tuple(x_.to(device) for x_ in torch.utils.data.dataloader.default_collate(x))) # This constructs batches of samples, and pre-emptily sends them to the device
    model.eval() # disable the model's dropout
    accuracy = []

    p = []
    t = []
    
    for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc="Evaluating"):

        with torch.no_grad(): # Don't keep track of the gradients for AutoGrad - we're not training
            out, bertOut, pooledOut, bert_hidden = model(input_ids, segment_ids, input_mask)
        
        # Transfer the predictions and targets to the CPU
        pred = out.detach().cpu().numpy()
        targ = label_ids.detach().cpu().numpy()

        # Compute the accuracy by taking the argmax of the model outputs, then comparing the target and predicted labels
        pred_s = np.argmax(pred, axis=1).flatten()
        accuracy.append( (pred_s==targ.flatten()).mean() )

        p += list(pred_s)
        t += list(targ.flatten())

    print(f"Accuracy: {mean(accuracy)}%")
    summary_writer.add_scalar('Accuracy', mean(accuracy), global_step)

    f1_micro = f1_score(t, p, average="micro")
    print(f"F1 Micro: {f1_micro}")
    summary_writer.add_scalar('F1 Score Micro', f1_micro, global_step)

    f1_macro = f1_score(t, p, average="macro")
    print(f"F1 Macro: {f1_macro}")
    summary_writer.add_scalar('F1 Score Macro', f1_macro, global_step)


    test['prediction'] = p
    test.to_excel('pred.xlsx')


#%%

for i in range(EPOCHS):
    epoch()
    eval(test_raw)




#%%

print("\n\n\n######################## Evaluation Performance ###########################\n")
eval(val_raw)

#%%