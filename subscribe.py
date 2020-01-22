import redis
import time


redis=redis.Redis(host="redis", port=6379)
tweetstream= redis.pubsub()
tweetstream.subscribe("stream")

if __name__ == '__main__':

    while True:
        message = tweetstream.get_message()
        if message:
            print("Tweets published")
            print(message["data"])

