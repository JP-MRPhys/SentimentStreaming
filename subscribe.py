import redis
import time
import json
#run this on python 3.6 pod


# just a simple subscribe to see if things are working as expected

config = json.load(open('./config.json'))
REDIS_HOST=config['redis_host']
REDIS_HOST_PORT=config['redis_host_port']
redis=redis.Redis(host=REDIS_HOST, port=REDIS_HOST_PORT)

tweetstream= redis.pubsub()
tweetstream.subscribe("stream")

if __name__ == '__main__':

    while True:
        message = tweetstream.get_message()
        if message:
            print("Tweets published")
            print(message["data"])

