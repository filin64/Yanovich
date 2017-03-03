from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas as pd
f = open ( 'housingprices.txt' , 'r')
data = [[float(i) for i in line.split(',')] for line in f]
plt.figure()
df = pd.DataFrame(data)
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
