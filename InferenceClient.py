# import spacy as spacy
from traits.trait_types import self
from tweepy import OAuthHandler, Stream, StreamListener

import pandas as pd
import json

import numpy as np
import tensorflow_hub as hub
import re
from bert import optimization, run_classifier, tokenization
import redis
import pickle
import time

import tensorflow as tf
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2

remove_char = ["!", "!", "@", "#", "$" ":", ")", ".", ";", ",", "?", "&", "http", "<"]

config = json.load(open('./config.json'))
DATA_COLUMN = config['data_column']
LABEL_COLUMN = config['label_column']

#twitter keys
consumer_key=config['twitter_consumer_key']
consumer_secret=config['twitter_consumer_secret']
access_key=config['twitter_access_key']
access_token_secret=config['twitter_token_secret']


#redis set up
REDIS_HOST=config['redis_host']
REDIS_HOST_PORT=config['redis_host_port']
redis=redis.Redis(host=REDIS_HOST, port=REDIS_HOST_PORT)


#bert set up
BERT_MODEL_HUB = config['bert_tf_hub']
tf.app.flags.DEFINE_string('server', config['bert_server'], 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS
channel = grpc.insecure_channel(FLAGS.server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = config['bert_model_name']
request.model_spec.signature_name = config['bert_model_signature']


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
        self.tweets = []
        self.model = []
        self.flip = False
        self.keywords = keywords
        self.keywords_re = "|".join(x for x in self.keywords)  # pattern for regex
        self.redis = redis
        self.tokenizer = tokenizer

    def on_data(self, data):
        # print("new tweet")
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
                data = {"tweet": tweet,
                        "returns": 1,
                        "filter": filter}

            return data

        return {"tweet": tweet, "returns": 1, "filter": "None"}

    def process_tweets(self, data):

        tweet = json.loads(data)
        if tweet["retweeted"]:
            return

        t = tweet["text"]
        t = self.clean_tweet(t, self.keywords_re)
        # print(t)
        # data={LABEL_COLUMN:1,DATA_COLUMN:t}


        self.tweets.append(t)
        if len(self.tweets) > 20:
            # TODO write to database (kafka or MongoBD)
            train = pd.DataFrame.from_dict(self.tweets)
            train=self.process_bert_results(train)
            self.send_to_redis(train)
            self.tweets = []

            redis.set("bert", "000")

        return

    def process_bert_results(self, train):
        """
        Tasks the results from bert services for sentiment analysis and provides out
        :return: sentiments; [batch_size] sentiment per tweet
        """

        redis.set("bert", "0001")

        # print(train[DATA_COLUMN])
        # print(train[LABEL_COLUMN])
        # train.apply(lambda x: print(x[DATA_COLUMN], x[LABEL_COLUMN]), axis=1)

        # Use the InputExample class from BERT's run_classifier code to create examples from the data
        train_InputExamples = train.apply(lambda x: run_classifier.InputExample(guid=None,
                                                                                # Globally unique ID for bookkeeping, unused in this example
                                                                                text_a=x[DATA_COLUMN],
                                                                                text_b=None,
                                                                                label=x[LABEL_COLUMN]), axis=1)
        # We'll set sequences to be at most 128 tokens long.
        MAX_SEQ_LENGTH = 150
        label_list = [0, 1]

        # Convert our train and test features to InputFeatures that BERT understands.
        train_features = run_classifier.convert_examples_to_features(train_InputExamples, label_list,
                                                                     MAX_SEQ_LENGTH, self.tokenizer)
        # self.tweets=[]

        input_ids = [x.input_ids for x in train_features]
        input_mask = [x.input_mask for x in train_features]
        segment_ids = [x.segment_ids for x in train_features]

        redis.set("bert", "test")

        batch_size = len(input_ids)

        start = time.time()

        request.inputs['input_ids'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_ids, shape=[batch_size, MAX_SEQ_LENGTH]))
        request.inputs['input_mask'].CopyFrom(
            tf.contrib.util.make_tensor_proto(input_mask, shape=[batch_size, MAX_SEQ_LENGTH]))
        request.inputs['segment_ids'].CopyFrom(
            tf.contrib.util.make_tensor_proto(segment_ids, shape=[batch_size, MAX_SEQ_LENGTH]))
        result = stub.Predict(request, 100.0)  # 10 secs timeout

        end = time.time()

        # ref: https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
        outputs_tensor_proto = result.outputs["predictated_labels"]
        shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
        outputs = np.array(outputs_tensor_proto.int_val).reshape(shape)

        # print(np.shape(outputs))
        # print("Request out is in time: {}".format(end - start))

        train["sentiment"] = outputs


        return train

    def on_error(self, status):
        # print(status)
        return

    def send_to_redis(self, results_dataframe):
        results_json = results_dataframe.to_json(orient="records")
        for count in (json.loads(results_json)):
            # if "filter" in count.keys():
            #   filter=count["filter"]
            #   redis.publish(filter, json.dumps(count))
            # else :
            redis.publish("stream", json.dumps(count))


if __name__ == '__main__':
    keywords = ['apple', 'google', 'tesla', 'SNP500']

    tokenizer = create_tokenizer_from_hub_module()

    l = StdOutListener(keywords=keywords, redis=redis, tokenizer=tokenizer)

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_token_secret)

    redis.set("foo", "bar2")
    stream = Stream(auth, l)
    stream.filter(track=keywords, languages=["en"])
    redis.set("foo", "bar4")