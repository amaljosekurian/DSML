import matplotlib.pyplot as plt

categories=["a","b","c","d"]
values=[1,2,3,4]

plt.bar(categories,values,color="green")
plt.xlabel(categories)
plt.ylabel(values)
plt.show()