import redis
import os
class RedisIndex:
    def __init__(self):
        self.client = redis.cluster.RedisCluster(host='10.158.176.47', port=6800)
        print(self.client.get_nodes())

    def exist(self,name, key):
        return self.client.hexists(name, key)


    def set(self, name, key, value):
        return self.client.hset(name, key, value)

    def get(self, name, key):
        return self.client.hget(name, key)

    def push(self, name, value):
        return self.client.lpush(name, value)

    def pop(self, name):
        return self.client.lpop(name)
