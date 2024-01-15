```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate random data for the histogram
data = np.random.randn(1000)

# Creating a customized histogram with a density plot
sns.histplot(data, bins=30, kde=True, color='lightgreen', edgecolor='red')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Customized Histogram with Density Plot')

# Display the plot
plt.show()
```