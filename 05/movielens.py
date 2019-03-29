from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate,train_test_split
import matplotlib.pyplot as plt

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)

# We retrieve predictions using our testset
predictions = algo.test(testset)

#We compile the delta between the real value and the 
# computed one. The prediction object is shaped like :
# (user: 196,item: 302,r_ui = 4.00,est = 4.06 ,{'actual_k': 40, 'was_impossible': False})
x = [elem[2] -elem[3] for elem in predictions]

plt.hist(x, 45, facecolor='g', alpha=1)

plt.xlabel('Deviation')
plt.ylabel('Occurences')
plt.title('Histogram of predictions deviation using the movielens database')
plt.axis([-3, 3, 0, 2000])
plt.grid(True)
plt.show()
