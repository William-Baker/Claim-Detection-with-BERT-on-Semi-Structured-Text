# INSTALL TWITTER API (!pip install TwitterAPI) and import necessary libraries/packages
# TwitterAPI Documentation here: https://github.com/geduldig/TwitterAPI/
from TwitterAPI import TwitterAPI, TwitterOAuth, TwitterRequestError, TwitterConnectionError, TwitterPager
import pandas as pd
import time
from os import makedirs, truncate
from copy import copy, deepcopy




class TwitterWrapper:
    """
    Note non-academic api keys are only permitted queries upto 512 characters, while academics are allowed 1024
    """
    class TweetFields:
        """Enumerates tweet fields: https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet"""
        attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,non_public_metrics,organic_metrics,possibly_sensitive,promoted_metrics,public_metrics,referenced_tweets,reply_settings,source,text,withheld='attachments','author_id','context_annotations','conversation_id','created_at','entities','geo','id','in_reply_to_user_id','lang','non_public_metrics','organic_metrics','possibly_sensitive','promoted_metrics','public_metrics','referenced_tweets','reply_settings','source','text','withheld'
        all_public = [attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,possibly_sensitive,public_metrics,referenced_tweets,reply_settings,source,text,withheld]
        all = [attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,non_public_metrics,organic_metrics,possibly_sensitive,promoted_metrics,public_metrics,referenced_tweets,reply_settings,source,text,withheld]
    class UserFields:
        """Enumerates twitter user fields: https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/user"""
        id,name,username,created_at,description,entities,location,pinned_tweet_id,profile_image_url,protected,public_metrics,url,verified,withheld='id','name','username','created_at','description','entities','location','pinned_tweet_id','profile_image_url','protected','public_metrics','url','verified','withheld'
        all = [id,name,username,created_at,description,entities,location,pinned_tweet_id,profile_image_url,protected,public_metrics,url,verified,withheld]

    class Search_Location:
        def __init__(self, locations: list[str]):
            """Search terms to add to a search query
            Args:
                locations (list[str]): List of country codes: 'US', 'CA', 'MX', ...
            """
            self.locations = locations
        def __str__(self):
            st = f" (place_country: {self.locations[0]}"
            if len(self.locations) > 1:
                for i in range(1, len(self.locations)):
                    st += f" OR place_country:{self.locations[i]}"
            return st + ")"

    def __init__(self, store_dir = '.store/'):
        consumer_key=         'UzQ5VzmIEaKnpwlTHElFCNXtF'
        consumer_secret=      '9ti9UAnJc9YSqvMPgFh3ezwiQ1LPVky3B2XjO93yTVVaYkMI3P'
        access_token_key=     '835273175357341697-jshWljSXBYHrV9RI3MMv39hGhuGHOww'
        access_token_secret=  'QqgcNHq7UlVxglTT6dIOnO5cMA39HmNtMKlxtuCJygUlP'
        self.api = TwitterAPI(consumer_key, consumer_secret, access_token_key, access_token_secret, api_version='2')
        # Bearer Token: AAAAAAAAAAAAAAAAAAAAAGFlVAEAAAAA7y3P1bfvJ%2BZfO%2FWjcLhJLoiMWeM%3DY8fER666XD4iE3Njx8MVJ4BHs1Op5OlrrCeeb7ML0JPuemBHOH


        self.store_dir = store_dir
        self.tweet_store = pd.DataFrame(columns=['Conversation Complete'])
        self.user_store = pd.DataFrame()
        
        makedirs(store_dir, exist_ok=True)
        # Attempt to read existing stores, if they exist
        try:
            self.tweet_store = pd.read_pickle(self.store_dir + 'tweets.pkl')
        except:
            pass
        try:
            self.user_store = pd.read_pickle(self.store_dir + 'users.pkl')
        except:
            pass


    @staticmethod
    def safe(lam):
        """Safeley execute api interactions"""
        try:
            res = lam()
        except TwitterRequestError as e:
            print(e.status_code)
            for msg in iter(e):
                print(msg)

        except TwitterConnectionError as e:
            print(e)

        except Exception as e:
            print(e)

        
        return res

    def get_user(self, username, user_fields=UserFields.all):
        joined_user_fields = ','.join(user_fields)

        user = TwitterWrapper.safe(
                    lambda: self.api.request(f'users/by/username/:{username}', {'user.fields': joined_user_fields})
        )
        user = user.json()['data']
        self.add_user_to_store(user)

        return user

    def get_recent_tweets(self, user_id, count, tweet_fields=TweetFields.all_public):
        joined_tweet_fields = ','.join(tweet_fields)
        
        tweets = TwitterWrapper.safe( 
                    lambda: self.api.request(f'users/:{user_id}/tweets', {'max_results': count, 'tweet.fields': joined_tweet_fields})
        )
        tweets = list(tweet for tweet in tweets.json()['data']) # parse the tweets into a dictionary immediately

        for tweet in tweets:
            self.add_to_tweet_store(tweet)

        return tweets
    
    
        

    def get_tweet(self, tweet_id, tweet_fields=TweetFields.all_public):
        """Retrieves the tweet corresponding to the given tweet_id, also functions for conversation_id's"""
        joined_tweet_fields = ",".join(tweet_fields)

        tweet = TwitterWrapper.safe( 
                    lambda:  self.api.request(f'tweets/:{tweet_id}', {'tweet.fields': joined_tweet_fields})
        )
        if tweet:
            tweet = tweet.json()['data'] # parse the tweet into a dictionary immediately
            self.add_to_tweet_store(copy(tweet))

        return tweet
    
    def mark_conversation_id_complete(self, conversation_id):
        if not 'Conversation Complete' in self.tweet_store.columns:
            self.tweet_store['Conversation Complete'] = False
        self.tweet_store['Conversation Complete'] = self.tweet_store.apply(lambda e: True if e['conversation_id'] == conversation_id else e['Conversation Complete'], axis=1)
        #self.tweet_store.to_pickle(self.store_dir + 'tweets.pkl')

    def conversation_id_done(self, conversation_id):
        this_conversaton = self.tweet_store[self.tweet_store['conversation_id'] == conversation_id]
        if this_conversaton.shape[0] > 0:
            if this_conversaton[this_conversaton['Conversation Complete'] == True].shape[0] > 0:
                return True
        return False
    
    def get_all_replies(self, conversation_id, tweet_fields=TweetFields.all_public):
        if self.conversation_id_done(conversation_id):
            return
        
        joined_tweet_fields = ",".join(tweet_fields)
        pager = TwitterPager(self.api, 'tweets/search/recent',
                            {'query': f'conversation_id:{conversation_id}', 'tweet.fields': joined_tweet_fields})
        
        self.get_all_replies_pager = pager.get_iterator()
        
        while True:
            try:
                tweet = next(self.get_all_replies_pager)
                print(f"Reply: {tweet['id']}: {tweet['text']}")
                self.add_to_tweet_store(copy(tweet))
            except StopIteration as e:
                self.mark_conversation_id_complete(conversation_id) # we're done now, so mark accordingly
                return
            except Exception as e:
                print(e)
                time.sleep(15 * 60)
                
        
        
        # for tweet in pager.get_iterator(wait=2):
        #     # tweet = tweet.json()['data'] # TODO why dont we need this!?
        #     self.add_to_tweet_store(copy(tweet))
        
        # self.mark_conversation_id_complete(conversation_id) # we're done now, so mark accordingly
        

    
    def search(self, search_term: str, count: int, tweet_fields=TweetFields.all_public):
        """
        Search using twitters search engine with text
        - can be a hashtag '#Space'
        - or: '#nowplaying (happy OR exciting OR excited OR favorite OR fav OR amazing OR lovely OR incredible) (place_country:US OR place_country:MX OR place_country:CA) -horrible -worst -sucks -bad -disappointing'
        https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
        max count 100
        """
        joined_tweet_fields = ",".join(tweet_fields)
        tweets = TwitterWrapper.safe(
            lambda: self.api.request('tweets/search/recent', {'query':search_term, 'tweet.fields':joined_tweet_fields, 'max_results': count})
        )
        tweets = list(tweet for tweet in tweets.json()['data']) # parse the tweets into a dictionary immediately
        for tweet in tweets:
            self.add_to_tweet_store(tweet)
        
        return tweets
    
    
    def search_sentiment(self, search_term: str, positive_sentiment: bool, count: int, tweet_fields=TweetFields.all_public):
        """
        Search using twitters search engine with text
        """
        joined_tweet_fields = ",".join(tweet_fields)
        tweets = TwitterWrapper.safe(
            lambda: self.api.request('tweets/search/recent', {'query':search_term, 'tweet.fields':joined_tweet_fields, 'max_results': count})
        )
        tweets = list(tweet for tweet in tweets.json()['data']) # parse the tweets into a dictionary immediately
        for tweet in tweets:
            self.add_to_tweet_store(tweet)
        
        return tweets
    
    def search_all_depth(self, search_term: str, tweet_fields=TweetFields.all_public):
        """
        Search using twitters search engine with text then get all the replies then carry on... forever
        - can be a hashtag '#Space'
        - or: '#nowplaying (happy OR exciting OR excited OR favorite OR fav OR amazing OR lovely OR incredible) (place_country:US OR place_country:MX OR place_country:CA) -horrible -worst -sucks -bad -disappointing'
        https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
        """
        joined_tweet_fields = ",".join(tweet_fields)
        
        
        pager = TwitterPager(self.api, 'tweets/search/recent', {'query':search_term, 'tweet.fields':joined_tweet_fields})
        
        self.search_all_depth_pager = pager.get_iterator(wait=2)
        
        while True:
            try:
                tweet = next(self.search_all_depth_pager)
                self.add_to_tweet_store(copy(tweet))
                print(f"{tweet['id']}: {tweet['text']}")
                
                if tweet['lang'] == 'en': # only bother if the original is english
                    self.get_all_replies(tweet['conversation_id'])
            except StopIteration as e:
                self.save_tweet_store()
                self.save_user_store()
                return
            except Exception as e:
                print(e)
                time.sleep(15*60)
                
    
    
    def get_recent_tweets_all(self, user_name, tweet_fields=TweetFields.all_public):
        joined_tweet_fields = ','.join(tweet_fields)
        
        user = self.get_user(user_name)
        user_id = user['id']
        
        for attempt in range(0,3):
            try:
                tweets = self.api.request(f'users/:{user_id}/tweets', {'max_results': 100, 'tweet.fields': joined_tweet_fields})
                break
            except Exception as e:
                print(e)
                time.sleep(15*60)
        
        tweets = list(tweet for tweet in tweets.json()['data']) # parse the tweets into a dictionary immediately

        for tweet in tweets:
            print(f"{tweet['id']}: {tweet['text']}")
            self.add_to_tweet_store(tweet)
            self.get_all_replies(tweet['conversation_id'])

    
    
    def get_location(self, lon, lat, p1, p2, count, tweet_fields=TweetFields.all_public):
        """
        Location should be 4 params, I'm not sure what they, are but this is my best guess
        """
        joined_tweet_fields = ','.join(tweet_fields)
        tweets = TwitterWrapper.safe( 
                    lambda: self.api.request('statuses/filter', {'locations':f"{lon},{lat},{p1},{p2}", 'max_results': count, 'tweet_feilds':joined_tweet_fields})
        )
        tweets = list(tweet for tweet in tweets.json()['data']) # parse the tweets into a dictionary immediately
        for tweet in tweets:
            self.add_to_tweet_store(tweet)
        
        return tweets
    

    def add_user_to_store(self, user):
        series = pd.Series(user, name=user['id'])
        self.user_store = self.user_store.append(series)
        if self.user_store.shape[0] % 10 == 0:
            self.save_user_store()
    
    def add_to_tweet_store(self,tweet):
        series = pd.Series(tweet, name=tweet['id'])
        self.tweet_store = self.tweet_store.append(series)
        if self.tweet_store.shape[0] % 1000 == 0:
            self.save_tweet_store()
    
    def save_tweet_store(self):
        self.tweet_store.to_pickle(self.store_dir + 'tweets.pkl')
    
    def save_user_store(self):
        self.user_store.to_pickle(self.store_dir + 'users.pkl')



