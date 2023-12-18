import numpy as np
def mat(matrix_name):
    row=int(input(f"enter the no of rows for {matrix_name}"))
    cols=int(input(f"enter the number of columns for {matrix_name}"))
    matrix=[]
    print("enter elements")
    for i in range(row):
        rows=[]
        for j in range(cols):
            element=int(input(f"enter the element for row{i+1}and{j+1}"))
            rows.append(element)
        matrix.append(rows)
    return np.array(matrix)

matrix1=mat("matrix1")
matrix2=mat("matrix2")
add=np.add(matrix1,matrix2)
print("sum of matrix",add)
diff=np.subtract(matrix1,matrix2)
print("difference b/w two matrix",diff)
prod=np.multiply(matrix1,matrix2)
print("product of two matrix",prod)
tran=np.transpose(matrix1)
print("transpose",tran)
d=np.dot(matrix1,matrix2)
print(d)