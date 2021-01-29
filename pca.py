from sklearn.decomposition import PCA

"""
主成分分析
"""
data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

transfer = PCA(n_components=0.95)

data_new = transfer.fit_transform(data)

print(data_new)
