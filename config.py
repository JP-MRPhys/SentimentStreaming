import json

config={ 'data_column' : "tweet",
         'label_column': "returns",
         'twitter_consumer_key': "2R4ARqSbMmjhtbv6NqL5tEvnr",
         'twitter_consumer_secret' : "BMgPg0aVGZGzM3elH8pKpsUB4BY3bfrBVzwC9iQ8qqyJ9KevCB",
         'twitter_access_key' : "99196167-O2u5HJ2WHhnHKCalrXZD5IlwrC3DbA5VofQ2lPJyq",
         'twitter_token_secret': "XLTHv1SWCKC2Q8QfAlF6ozToxULBdqcpbmZBcdAwYtXnB",
         'redis_host': "redis",
         'redis_host_port': 6379,
         'spacy_model':"en_core_web_sm",
         'bert_tf_hub': "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
         'bert_server': "bertservice:8500",
         'bert_model_name': "bert",
         'bert_model_signature': "bert_predictions"
}

if __name__ == '__main__':

    json.dump(config, open('./config.json', 'w'))
    data=json.load(open('./config.json'))
    print(data['bert_model_signature'])



