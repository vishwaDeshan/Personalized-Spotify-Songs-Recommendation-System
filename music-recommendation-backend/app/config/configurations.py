from pymongo.mongo_client import MongoClient
import certifi

# MongoDB URI
uri = "mongodb+srv://vishwawaweliyadda1998:CAy9qlGIyA0HYVS6@nexawavecluster.e03zg.mongodb.net/?retryWrites=true&w=majority&appName=NexaWaveCluster"
client = MongoClient(uri, tlsCAFile=certifi.where())

db = client.user_db  
collection = db["user_data"]