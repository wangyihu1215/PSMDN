from matplotlib import pyplot as plt


def plot_curve(data, name):
    figure = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['loss'], loc='upper right')
    plt.title('Training Process')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(r'./figure/'+name+'.png')
    plt.show()
