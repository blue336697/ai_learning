import matplotlib.pyplot as plt
from IPython.display import display  # Jupyter display function
from machinelearn.common.plt_overfit import overfit_example, output



plt.close("all")
display(output)
ofit = overfit_example(False)
plt.show()

