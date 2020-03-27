import numpy as np

arr1=np.array([1,2,3])
arr2=np.array([[1,2,3],[4,5,6]])
print(arr1[:,np.newaxis],[row for row in arr1])
print(arr2[:,np.newaxis],[row for row in arr2])