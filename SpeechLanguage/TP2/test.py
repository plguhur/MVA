from utils import *
import os

data = "data"
path = os.path.join(data, os.listdir(data)[0])

db = Dataset(path)
trainset = db.get_train()


