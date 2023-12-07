import matplotlib.pyplot as plt

def plot_graph(data, params,xlabel,ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(data,label=params["label"])
    plt.title(params["title"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()