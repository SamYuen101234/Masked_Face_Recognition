import pymongo
from pymongo import MongoClient

def connect(url):
    try:
        client = MongoClient(url)
        print("Connceted successfully")
        show_all_db(client)
        return client
    except pymongo.errors.ServerSelectionTimeoutError as err:
        print(err)

def show_all_db(client):
    print("All database:",client.list_database_names())

def get_db(client, db_name):
    db = client[db_name]
    show_all_col(db)
    return db

def show_all_col(db):
    print("All collections:", db.list_collection_names())

def get_col(db, col_name):
    collist = db.list_collection_names()
    if col_name in collist:
        return db[col_name]
    else:
        print(col_name,'does not exist.')

def create_col(db, col_name):
    db[col_name]

def drop_col(col):
    col.drop()

def insert(col,list):
    col.insert_many(list)

def delete(col, query):
    x = col.delete_many(query)
    print(x.deleted_count, " documents deleted.")

def update(col, query, newvalue):
    x = col.update_many(query, newvalue)
    print(x.modified_count, "documents updated.")

def find(col, query):
    return col.find(query)

def show_doc(doc):
    for x in doc:
        print(x)


client= connect("mongodb+srv://user1:iUUuNtJEWS8niQyj@fypdata.qk5zn.mongodb.net/")
db = get_db(client, 'fyp')
user_col = get_col(db, 'user')
list = [{'id':1234, 'name':'sam'}]
#insert(user_col, list)
#update(user_col, { "name": "hello" }, { "$set": { "name": "sam" } })
#delete(user_col, { "name": "sam" })
doc = find(user_col, { "name": "sam" })
show_doc(doc)