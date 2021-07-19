from os import close
import config as cfg
import pymongo

# No Sql database which will be used to store the users and projects data
client = pymongo.MongoClient(cfg.db_conn_str, serverSelectionTimeoutMS=5000)
db = client.github_project
users = db.users_3
repositories = db.repositories_3

close = client.close