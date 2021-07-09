"""
Script for test.
"""
import pickle

path_X = "./pickles/balance/2-feature-model-0-X.pkl"

with open(path_X, "rb") as f:
    feature_X = pickle.load(f)


print("len(feature_X):", len(feature_X))

# # calculate acc
# count = 0
# for x, y in zip(feature_X, target_y):
#     if x == y:
#         count += 1

# print("acc:", count / len(target_y))
