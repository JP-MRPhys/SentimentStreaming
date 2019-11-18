import redis
import time
import pickle
import json

redis=redis.Redis(host="127.0.0.1", port=6379)
tweetstream= redis.pubsub()
tweetstream.subscribe(["apple", "google"])

if __name__ == '__main__':
    #redis.set("test", "hi")

    while True:

        message = tweetstream.get_message()
        if message:
            print("Tweets published")

            print(type(message["data"]))
            print(message.keys())

            if type(message["data"]) is bytes:
                mess=json.loads(message["data"])
                print(mess)

        time.sleep(1)