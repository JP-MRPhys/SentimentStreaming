import redis
import time
import pickle

redis=redis.Redis(host="127.0.0.1", port=6379)
tweetstream= redis.pubsub()
tweetstream.subscribe("google")

if __name__ == '__main__':
    #redis.set("test", "hi")

    while True:
        message = tweetstream.get_message()
        if message:
            print("Stocks published")

            data=pickle.loads(message)

            print(data)

        time.sleep(1)