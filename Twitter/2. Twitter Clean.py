#%%
import pandas as pd
import re


#%% =========================== Clean Twitter Data ===============================================================
# Replacing hastags, mentions, URLs and retweets with binary tokens

tweets = pd.read_pickle('../datasets/tweets.pkl')

# Remove tweets with duplicate text - indicates it's a bot
tweets = pd.DataFrame(tweets[~tweets['text'].duplicated(False)])

# To lower case
tweets['text'] = tweets['text'].apply(lambda text: text.lower())

# Replace URLs with [URL] binary text
url_re = re.compile("""(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})""")
tweets['text'] = tweets['text'].apply(lambda text: url_re.sub("[URL]", text))



# Replace hashtags with [TAG] followed by the hashtag, e.g. #peace becomes [TAG] peace
hashtag_regex = re.compile("""#[^ !@#$%^&*(),.?":{}|<>]*""")

def process_hashtags(text):
    matches = list(hashtag_regex.finditer(text))
    while len(matches):
        start, end = matches[0].span()
        before = text[:start]
        tag = text[start:end]
        after = text[end:]
        text = before + "[TAG] " + tag[1:] + after
        matches = list(hashtag_regex.finditer(text))
    return text

tweets['text'] = tweets['text'].apply(process_hashtags)



# Retweets start with rt, raplace starting rt with [RT] tag
def process_retweets(text):
    if text[0:2] == 'rt':
        return '[RT]' + text[2:]
    else:
        return text
tweets['text'] = tweets['text'].apply(process_retweets)

# Replace mentions with [MEN] e.g. @Bob
mention_regex = re.compile("""@[^ !@#$%^&*(),.?":{}|<>]*""")
tweets['text'] = tweets['text'].apply(lambda text: mention_regex.sub("[MEN]", text))



# Keep only english tweets
tweets = pd.DataFrame(tweets[tweets['lang'] == 'en'])


# Remove tweets again, some bots use random urls to obscure duplicate tweets
tweets = pd.DataFrame(tweets[~tweets['text'].duplicated(False)])


# Ensure no empty tweets before proceeding
tweets = pd.DataFrame(tweets[tweets['text'] != ""])




#%% ====================== Replace Unicode and ASCII emoji's with thier descriptions =====================================================

import re
from emoji import UNICODE_EMOJI

ALL_EMOJIS = UNICODE_EMOJI['en'] # gives us a dictionary mapping emojis to english text
# replace underscores in emoji dictionary
for emoji, translation in ALL_EMOJIS.items():
    ALL_EMOJIS[emoji] = translation.replace('_', ' ').replace(':', ' ').lower()

        


ASCII_MAPPING = [
    ([':‑)',':)',':-]',':]',':->',':>','8-)','8)',':-}',':}',':o)',':c)',':^)','=]','=)'], "happy face"),
    ([':‑D', ':D', '8‑D', '8D', '=D', '=3', 'B^D', 'c:', 'C:', ':-))'], "big grin face"),
    (['x‑D', 'xD', 'X‑D', 'XD'], "laughing face"),
    ([':‑(',':(',':‑c',':c',':‑<',':<',':‑[',':[',':-||','>:[',':{',':@',':(',';('], "sad face"),
    ([':‑(', ":'‑(", ":'(", ':=('], "crying face"),
    ([":'‑)", ":')", ':"D'], "happy tears face"),
    (["D‑':", 'D:<', 'D:', 'D8', 'D;', 'D=', 'DX'], "great dismay face"),
    ([':‑O', ':O', ':‑o', ':o', ':-0', '8‑0', '>:O', '=O', '=o', '=0'], "suprose face"),
    ([':-3', ':3', '=3', 'x3', 'X3  >:3'], "cat face"),
    ([':-*', ':*', ':×'], "kiss face"),
    (['‑)', ';)', '*-)', '*)', ';‑]', ';]', ';^)', ';>', ':‑,', ';D', ';3'], "wink face"),
    ([':‑P', ':P', 'X‑P', 'XP', 'x‑p', 'xp', ':‑p', ':p', ':‑Þ', ':Þ', ':‑þ', ':þ', ':‑b', ':b', 'd:', '=p', '>:P'], "tongue face"),
    ([':-/', ':/', ':‑.', '>:\\', '>:/', ':\\', '=/', '=\\', ':L', '=L', ':S'], "skeptical face"),
    ([':‑|', ':|'], "straight face"),
    ([':$', '://)', '://3'], "embarrased face"),
    ([':‑X', ':X', ':‑#', ':#', ':‑&', ':&'], "sealed lips face"),
    (['O:‑)', 'O:)', '0:‑3', '0:3', '0:‑)', '0:)', '0;^)'], "angel face"),
    (['>:‑)', '>:)', '}:‑)', '}:)', '3:‑)', '3:)', '>;‑)', '>;)', '>:3', ';3'], "evil face")
    ]
ASCII_DICT = dict()
for faces, desc in ASCII_MAPPING:
    for face in faces:
        ASCII_DICT[face] = desc

ALL_EMOJIS = ALL_EMOJIS | ASCII_DICT




# A very expensive method of replacement, but simple and robust, since the code need only be ran once this is acceptable
from tqdm import tqdm
for emoji, desc in tqdm(ALL_EMOJIS.items(), "Replacing Emojis with their descriptions"):
    tweets['text'] = tweets['text'].apply(lambda s: s.replace(emoji, desc))




# Save the tweets
tweets.to_pickle('../datasets/cleaned tweets.pkl')



#%% ============================== Train the word peice tokenizer ================================================================

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)

# store plaintext in file
temp_text_output = 'temp_tweets.txt'
with open(temp_text_output, 'w') as f:
    dfAsString = tweets['text'].apply(lambda x: x.replace('\n', ' ')).tolist()
    dfAsString = '\n'.join(dfAsString)
    f.write(dfAsString)



tokenizer.train([temp_text_output], vocab_size=30522, min_frequency=2,
                limit_alphabet=1000, wordpieces_prefix='##',
                special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[URL]', '[RT]', '[TAG]', '[MEN]'])

import os
os.remove(temp_text_output)

os.makedirs('../bert-it', exist_ok=True)

if os.path.isfile('../bert-it/bert-it-vocab.txt'):
    os.remove('../bert-it/bert-it-vocab.txt')

tokenizer.save_model('../bert-it', 'bert-it')








#%%