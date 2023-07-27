import pickle

res = pickle.load(open('./runs/20230515_192008/results.pkl', 'rb'))

res[5]['model'].best_iteration