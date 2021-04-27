"""
Script for test.
"""
import pickle

#path_X = './pickles/balance/model-0-X.pkl'
path_y = './pickles/balance/model-0-y.pkl'
path_y_2 = './pickles/balance/model-1-y.pkl'
path_y_3 = './pickles/balance/model-2-y.pkl'
path_y_4 = './pickles/balance/model-3-y.pkl'
path_y_5 = './pickles/balance/model-4-y.pkl'

# with open(path_X, 'rb') as f:
#     feature_X = pickle.load(f)
with open(path_y, 'rb') as f:
    feature_y = pickle.load(f)

with open(path_y_2, 'rb') as f:
    feature_y_2 = pickle.load(f)

with open(path_y_3, 'rb') as f:
    feature_y_3 = pickle.load(f)

with open(path_y_4, 'rb') as f:
    feature_y_4 = pickle.load(f)

with open(path_y_5, 'rb') as f:
    feature_y_5 = pickle.load(f)

print(feature_y == feature_y_2 == feature_y_3 == feature_y_4 == feature_y_5)
# print(len(feature_X))
print(len(feature_y))

count = 0

# for x,y in zip(feature_X, feature_y):
#     if x == y:
#         count += 1
# for x,y in zip(feature_y, feature_y_2):
#     if x == y:
#         count += 1

# print('acc:',count/len(feature_y))
