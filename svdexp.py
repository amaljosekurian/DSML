import numpy as np
matrix=np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])

U,S,VT=np.linalg.svd(matrix)
print("matrix U")
print(U)
print("Matrix S")
print(np.diag(S))
print("matrix VT")
print(VT)

print("old matrix")
old=np.dot(U,np.dot(np.diag(S),VT))
print(old)
