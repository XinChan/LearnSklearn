from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# knn = KNeighborsClassifier(n_neighbors = 5)
# # model = knn.fit(X_train, y_train)
# # print(model.score(X_test,y_test))

# scores = cross_val_score(knn, X, y, cv = 5, scoring = 'accuracy')
# # print(scores) 
# # [0.96666667 1.         0.93333333 0.96666667 1.        ]

# print(scores.mean())


# k_ranges = range(1,31)
# k_scores = []
# for k in k_ranges:
# 	knn = KNeighborsClassifier(n_neighbors = k)
# 	scores = cross_val_score(knn, X, y,cv = 10, scoring ='accuracy')
# 	k_scores.append(scores.mean())

# plt.plot(k_ranges, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross_validated Accuracy')
# plt.show()



k_ranges = range(1,31)
k_scores = []
for k in k_ranges:
	knn = KNeighborsClassifier(n_neighbors = k)
	loss = -cross_val_score(knn, X, y,cv = 10, scoring ='neg_mean_squared_error')
	k_scores.append(loss.mean())

plt.plot(k_ranges, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross_validated Accuracy')
plt.show()