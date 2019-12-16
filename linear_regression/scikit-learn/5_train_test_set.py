import numpy as np
from sklearn.model_selection import train_test_split

a = np.arange(1, 101)
print(a)
b = np.arange(501, 601)
split = train_test_split(a, test_size=0.2, random_state=42)    # random_state is the same as np.random.seed(), bring the same set of numbers
# print(split)
print(split[0].shape, split[1].shape)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=-.2, random_state=42)
print('Train split\n', a_train, a_train.shape, '\n', b_train, b_train.shape, '\n', a_test, a_test.shape, '\nTest split', b_test, b_test.shape)