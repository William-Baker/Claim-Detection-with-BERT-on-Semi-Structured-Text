
#%%

import sys
sys.path.insert(0, '../') # add to the parent directory to the path, so we can import the package

from TwitterAPIWrapper import TwitterWrapper
import pandas as pd

tw = TwitterWrapper()




# Import Plaintext Dataset
df = pd.read_table('dataset.txt')

df['tweet_id'] = df['tweet_id'].astype(str)

for id in df['tweet_id'].tolist():
    try:
        tw.get_tweet(id)
    except KeyError:
        pass


df.set_index('tweet_id', inplace=True)

s = df.join(tw.tweet_store)


s.to_pickle('Wenjia labelled tweets.pkl')
s.to_excel('Wenjia labelled tweets.xlsx')


#%%

