import matplotlib.pyplot as plt

def plot_graph(data, title,label,xlabel,ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(data[0],label="alpha="+str(label[0]))
    plt.plot(data[1],label="alpha="+str(label[1]))
    plt.plot(data[2],label="alpha="+str(label[2]))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(title)
    # plt.show()