#%%
import pandas as pd

tweets = pd.read_pickle('../datasets/cleaned tweets.pkl') # read the cleaned tweets


tweets['acknowledgments'] = tweets['public_metrics'].apply(lambda x: sum(x.values()))
tweets = tweets[tweets['acknowledgments'] > 0]


tweets = tweets[tweets['text'].apply(lambda x: len(x) > 50)]

tweets[['text']].sample(frac=1).to_excel('../datasets/labelled tweets.xlsx')

#%%


# Tweets are labelled now, under a new column named 'claim'


#%%
import pandas as pd

my_labelled_tweets = pd.read_excel('../datasets/labelled tweets.xlsx')
wenjia_labelled_tweets = pd.read_pickle('Wenjia Ma et al/Wenjia labelled tweets.pkl')

#%%

my_labelled_tweets.set_index(my_labelled_tweets.columns[0], inplace=True) # Importing from excel creates a new index, go back to using the first column
labelled_tweets = pd.concat([my_labelled_tweets, wenjia_labelled_tweets])
labelled_tweets.dropna(subset=['claim', 'text'], inplace=True) # Remove any tweets that haven't been annotated


def map_labels(label):
    if label == 0 or label == 1:
        return label
    else:
        label = str(label).lower()
        mapping = {'y':1, 'yes':1, 'true':1, 't':1, 'n':0, 'no':0, 'false':0, 'f':0}
        if label not in mapping:
            error_message = f"ERROR: unrecognized claim label found \"{label}\" expected any of {mapping.keys()}"
            print(error_message)
            raise Exception(error_message)
        else:
            return mapping[label]
            

labelled_tweets['claim'] = labelled_tweets['claim'].apply(map_labels)

#%%

claims = labelled_tweets[labelled_tweets['claim'] == 1]
non_claims = labelled_tweets[labelled_tweets['claim'] == 0]


claim_count = claims.shape[0]
non_claim_count = non_claims.shape[0]

# randomly select the same number of claims from each class type, then randomise the order of the result
sample_size = min(claim_count, non_claim_count)
balanced_labelled_tweets = pd.concat([claims.sample(sample_size), non_claims.sample(sample_size)]).sample(frac=1) 

balanced_labelled_tweets.rename(columns={'claim':'y', 'text':'x'}, inplace=True)

# mark 13% as the test set and 7% as the validation set

test_index = int(balanced_labelled_tweets.shape[0] * 0.8)
val_index = int(balanced_labelled_tweets.shape[0] * 0.93)

train_set = pd.DataFrame(balanced_labelled_tweets.iloc[:test_index])
test_set = pd.DataFrame(balanced_labelled_tweets.iloc[test_index:val_index])
val_set = pd.DataFrame(balanced_labelled_tweets.iloc[val_index:])

train_set['split'] = 'train'
test_set['split'] = 'test'
val_set['split'] = 'val'

balanced_labelled_tweets = pd.concat([train_set, test_set, val_set])

balanced_labelled_tweets.to_pickle('../datasets/labelled tweets.pkl')

