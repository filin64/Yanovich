from pandas.tools.plotting import scatter_matrix
f = open ( 'housingprices.txt' , 'r')
data = [[float(i) for i in line.split(',')] for line in f]
print (data)