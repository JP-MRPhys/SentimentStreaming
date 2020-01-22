#import spacy as spacy
from traits.trait_types import self
from tweepy import OAuthHandler, Stream, StreamListener
import tensorflow as tf
import pandas as pd
import json
#import spacy
import numpy as np
import tensorflow_hub as hub
import re
from bert import optimization, run_classifier, tokenization
import redis
import pickle
#redis
#redis=redis.Redis(host="127.0.0.1", port=6379)

DATA_COLUMN = 'tweet'
LABEL_COLUMN = 'returns'

#nlp = spacy.load('en_core_web_sm')
consumer_key="2R4ARqSbMmjhtbv6NqL5tEvnr"
consumer_secret="BMgPg0aVGZGzM3elH8pKpsUB4BY3bfrBVzwC9iQ8qqyJ9KevCB"
access_key="99196167-O2u5HJ2WHhnHKCalrXZD5IlwrC3DbA5VofQ2lPJyq"
access_token_secret="XLTHv1SWCKC2Q8QfAlF6ozToxULBdqcpbmZBcdAwYtXnB"
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
remove_char=["!", "!", "@", "#", "$" ":",")", "." , ";" ,",","?", "&", "http", "<"]
REDIS_HOST="127.0.0.1"
REDIS_HOST_PORT=6329

tf.app.flags.DEFINE_string('server', '35.232.105.219:8500', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    def __init__(self, keywords, redis, tokenizer):
       self.tweets=[]
       self.model=[]
       self.flip=False
       self.keywords=keywords
       self.keywords_re = "|".join(x for x in self.keywords) #pattern for regex
       self.redis=redis
       self.tokenizer=tokenizer


    def on_data(self, data):
        #print("new tweet")
        self.process_tweets(data)
        return True

    def on_status(self, status):
        if status.retweeted_status:
            return

    def clean_tweet(self, tweet, keywords_re):


        tweet = re.sub('http\S+\s*', '', tweet)  # remove URLs
        tweet = re.sub('RT|cc', '', tweet)  # remove RT and cc
        tweet = re.sub('#\S+', '', tweet)  # remove hashtags
        tweet = re.sub('@\S+', '', tweet)  # remove mentions
        tweet = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', tweet)
        # remove punctuations
        tweet = re.sub('\s+', ' ', tweet)  # remove extra whitespace
        tweet = re.sub('^\s', '', tweet)  # remove extra whitespace

        filters = re.findall(keywords_re, tweet)

        # TODO: Many tweets contains do not contain filter words the documentation to get all the tweets or filter info:
        # TODO: This may be a paid feature via twitter

        if (filters):
            for filter in filters:

                data={"tweet": tweet,
                      "returns": 1,
                      "filter": filter}

                s=json.dumps(data)
                #redis.publish(filter,s)
                print(data)

        return tweet

    def process_tweets(self, data):

        tweet=json.loads(data)
        if  tweet["retweeted"]:
            return

        t=tweet["text"]
        t=self.clean_tweet(t, self.keywords_re)
        #print(t)

        data={LABEL_COLUMN:1,DATA_COLUMN:t}
        label_list=[0,1]

        self.tweets.append(data)
        if len(self.tweets)>20:

            # TODO write to database (kafka or MongoBD)
            train = pd.DataFrame.from_dict(self.tweets)
            #print(train[DATA_COLUMN])
            #print(train[LABEL_COLUMN])
            #train.apply(lambda x: print(x[DATA_COLUMN], x[LABEL_COLUMN]), axis=1)

            # Use the InputExample class from BERT's run_classifier code to create examples from the data
            train_InputExamples = train.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                                    # Globally unique ID for bookkeeping, unused in this example
                                                                                    text_a=x[DATA_COLUMN],
                                                                                    text_b=None,
                                                                                    label=x[LABEL_COLUMN]), axis=1)
            # We'll set sequences to be at most 128 tokens long.
            MAX_SEQ_LENGTH = 150

            # Convert our train and test features to InputFeatures that BERT understands.
            train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, self.tokenizer)
            #self.tweets=[]

        #d = nlp(a["text"])
        #for token in d:
        #    token.vector

        return

    def on_error(self, status):
        print(status)

    def clean_tweet_old(self, tweet, keywords):

        #this should be made non blocking

        clean_tweet = " ".join([word for word in tweet.split()
                                if word not in remove_char and '@' not in word and '<' not in word])

        t = re.sub("[!@#$:).;,?&]", " ", clean_tweet)

        filters = re.findall(keywords, t, flags=re.IGNORECASE)
        #change to lowercase
        filters = map(str.lower, filters)

        #remove duplicates
        filters=list(dict.fromkeys(filters))

        # TODO: filter words are not avaliable read the documentation to get all the tweets
        if (filters):
            print("tweet contains follow key words")

            print(filters)
            for filter in filters:
                #redis.publish(filter,t)
                print("published tweet" + t)

        return

if __name__ == '__main__':


    keywords=['apple', 'google', 'tesla', 'SNP500']

    tokenizer = create_tokenizer_from_hub_module()

    l = StdOutListener(keywords=keywords, redis=redis, tokenizer=tokenizer)

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=keywords, languages=["en"])

