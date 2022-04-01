
#%%

from TwitterAPIWrapper import TwitterWrapper


tw = TwitterWrapper()



#%% ================================================================= NASA Example ===========================================================================


nasa_id = tw.get_user('NASA')['id']
nasa_tweets = tw.get_recent_tweets(nasa_id, 5)
tweet_id = nasa_tweets[3]['id']
tweet = tw.get_tweet(tweet_id)
conversation_id = tweet['conversation_id']
root = tw.get_tweet(conversation_id)

#%%

tw.search_all_depth('#unpopularopinion')
tw.search_all_depth('#discussion')

#%%

# Top movie franchises
tw.search_all_depth('#HarryPotter')
tw.search_all_depth('#StarWars')
tw.search_all_depth('#Marvel')
tw.search_all_depth('#LOTR')
tw.search_all_depth('#JamesBond')
tw.search_all_depth('#XMen')
tw.search_all_depth('#DC')

# Top book franchises
tw.search_all_depth('#Goosebumps')
tw.search_all_depth('#Narnia')
tw.search_all_depth('#CSLewis')
tw.search_all_depth('#LionWitchWardrobe')
tw.search_all_depth('#Dickens')
tw.search_all_depth('#CharlesDickens')
tw.search_all_depth('#AgathaChristie')
#%%

tw.search_all_depth('#netflix')

#%%

tw.search_all_depth('#newrelease')

#%%

for u in ['primevideo', 'waterstones', 'nasa', 'imdb', 'warnerbros', 'barackobama', 'disney', 'rottentomatoes', 'sonypictures', 'bbcnews', 'netflix', 'elonmusk', 'amazon', 'youtube']:
    tw.get_recent_tweets_all(u)

#%%

# Top movie franchises
tw.search_all_depth('HarryPotter')
tw.search_all_depth('StarWars')

#%%

tw.search_all_depth('Marvel')
tw.search_all_depth('LOTR')
tw.search_all_depth('JamesBond')
tw.search_all_depth('XMen')
tw.search_all_depth('DC')

# Top book franchises
tw.search_all_depth('Goosebumps')
tw.search_all_depth('Narnia')
tw.search_all_depth('CSLewis')
tw.search_all_depth('LionWitchWardrobe')
tw.search_all_depth('Dickens')
tw.search_all_depth('CharlesDickens')
tw.search_all_depth('AgathaChristie')

tw.search_all_depth('netflix')

tw.search_all_depth('newrelease')

#%%
for u in ['amazon', 'youtube']:
    tw.get_recent_tweets_all(u)


#%%

