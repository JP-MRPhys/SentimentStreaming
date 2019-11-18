import redis
import time

redis=redis.Redis(host="127.0.0.1", port=6379)
tweetstream= redis.pubsub()
tweetstream.subscribe("apple")

if __name__ == '__main__':
    #redis.set("test", "hi")

    while True:
        message = tweetstream.get_message()
        if message:
            print("Tweets published")
            print(message["data"])

        time.sleep(1)