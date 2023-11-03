import json
import os
import pickle
from matplotlib import pyplot as plt


def read_json(path):
    if os.path.exists(path):
        f = open(path, 'r')
        contents = f.read()
        data = json.load(contents)
    else:
        print('no such file:{}'.format(path))
    return data


def save_json(path, data: dict):
    b = json.dumps(data)
    f = open(path, 'w')
    f.write(b)
    f.close()
    print('success saving file:{}'.format(path))


def save_pickle(data, filename):
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()


def read_pickle(filename):
    # reload a file to a variable
    with open(filename, 'rb') as file:
        a_dict1 = pickle.load(file)
    return a_dict1


def config_ax(ax, xylabels=None, title=None, loc=None):
    """
    Configure appearance of the Matplotlib figure using given axis.
    """

    ax.grid(True, color='0.9')
    ax.set_frame_on(False)
    ax.tick_params(color='0.9')

    if xylabels is not None:
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])

    if title is not None:
        ax.set_title(title)

    if loc is not None:
        ax.legend(loc=loc)


def plot_one_y(x,y,label,xylabels,save_path):
    fig, ax = plt.subplots(tight_layout=True)
    ax.plot(x, y, label=label)
    config_ax(ax, xylabels=xylabels, loc='best')
    plt.savefig(save_path)
