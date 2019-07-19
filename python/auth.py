import os
import json

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "auth.json"), "r") as f:
    cfg = json.load(f)

# mariadb
def get_db_host():     return cfg["db"]["host"]
def get_db_user():     return cfg["db"]["user"]
def get_db_password(): return cfg["db"]["password"]
def get_db_name():     return cfg["db"]["db"]

def store_name(): return cfg["storage"]["name"]
def store_key():  return cfg["storage"]["key"]

