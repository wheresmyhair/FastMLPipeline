import os
import pickle
from typing import Union
from cryptography.fernet import Fernet


os.environ['ENCRYPTION_KEY'] = Fernet.generate_key().decode()
'''
Generate a key using Fernet.generate_key(), then store on environment variable or file.
Such as 

# os.environ['ENCRYPTION_KEY'] = Fernet.generate_key().decode()

or

with open('my_secret.key', 'wb') as f:
    f.write(Fernet.generate_key())
os.environ['ENCRYPTION_KEY_PATH'] = 'my_secret.key'
'''


def load_key(fp: str = None) -> Union[bytes, None]:
    if fp:
        with open(fp, 'rb') as f:
            return f.read()
    if key := os.getenv('ENCRYPTION_KEY'):
        return key.encode()


def load_pickle(fp: str) -> object:
    if key := load_key(os.getenv('ENCRPYTION_KEY_PATH')):
        fernet = Fernet(key)
        with open(fp, 'rb') as f:
            return pickle.loads(fernet.decrypt(f.read()))
    with open(fp, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: object, fp: str):
    if key := load_key(os.getenv('ENCRPYTION_KEY_PATH')):
        fernet = Fernet(key)
        with open(fp, 'wb') as f:
            f.write(fernet.encrypt(pickle.dumps(obj)))
        return
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)
        
