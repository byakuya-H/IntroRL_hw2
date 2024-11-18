from typing import Any, List, Union
from collections import OrderedDict
import io, sys, time, os, pickle
from base64 import standard_b64encode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.misc


class table(OrderedDict):
    def __getattr__(self, attr) -> Any:
        return self.get(attr, None)

    def __setattr__(self, attr: str, value) -> None:
        self[attr] = value


def merge_conf(conf1: dict, conf2: dict) -> table:
    return table({**conf2, **conf1})


class _Drawer_kitty:
    clear_screen = b"\x1b_Ga=d\x1b\\"

    # def __init__(self):
    #     # os.system("clear" if os.name == "posix" else "cls")
    #     self._w(b"\x1b[100T")

    def draw(self, obs: np.ndarray):
        time.sleep(0.25)
        self._w(self.clear_screen)
        buf = io.BytesIO()
        plt.imsave(buf, obs, format="png")
        data = standard_b64encode(buf.getvalue())
        im = b""
        while data:
            chunk, data = data[:4096], data[4096:]
            m = 1 if data else 0
            im = b"\x1b_G"
            im += (f"m={m}" + ",a=T,f=100,c=50,C=1,X=50,Y=50").encode("ascii")
            im += b";" + chunk if chunk else b""
            im += b"\x1b\\"
        self._w(im)

    def _w(self, im):
        sys.stdout.buffer.write(im)
        sys.stdout.flush()

    def __del__(self):
        self._w(self.clear_screen)


class _Drawer_plt:
    def __init__(self, plt):
        self.plt = plt
        self.plt.ion()
        self.plt.show(block=False)

    def draw(self, obs: np.ndarray):
        self.plt.clf()
        with self.plt.ion():
            self.plt.imshow(obs)
            self.plt.draw()
            self.plt.pause(0.15)

    def __del__(self):
        self.plt.clf()
        self.plt.cla()
        self.plt.ioff()


class Drawer:
    def __init__(self, plt, method: str = "plt"):
        if method == "kitty":
            self.drawer = _Drawer_kitty()
        else:
            self.drawer = _Drawer_plt(plt)

    def draw(self, obs: np.ndarray):
        info = np.zeros(shape=obs.shape[1:] + (3,))
        info[:, :, 0], info[:, :, 1], info[:, :, 2] = (
            obs[0, :, :],
            obs[1, :, :],
            obs[2, :, :],
        )
        obs = info
        self.drawer.draw(obs)

    def __del__(self):
        del self.drawer


def _save_imgs(imgs: List[np.ndarray], dir: str):
    for i in range(len(imgs)):
        scipy.misc.toimage(imgs[i], cmin=0.0, cmax=1).save(
            os.path.join(dir, str(i) + ".png")
        )


def _save_model(model: dict, file):
    with open(file, "wb") as f:
        pickle.dump(model, f)


def save_result(
    save_dir,
    imgs: Union[List[np.ndarray], None] = None,
    Qmodel: Union[dict, None] = None,
):
    if imgs and Qmodel is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    saved_res = os.listdir(save_dir)
    subdir = time.strftime("%D_%H:%M:%S", time.localtime(time.time())).replace("/", "-")
    if subdir in saved_res:
        subdir = subdir + str(len([0 for i in saved_res if i == subdir]))
    save_dir = os.path.join(save_dir, subdir)
    os.makedirs(save_dir, exist_ok=True)
    if imgs is not None:
        _save_imgs(imgs, save_dir)
    if Qmodel is not None:
        _save_model(Qmodel, os.path.join(save_dir, "Qmodel.pkl"))

def plot_fig(x, ct, m, ma, mi):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(x, ct, color="red", label="converge time")
    ax.plot(x, m, color="blue", label="score")
    ax.fill_between(x, mi, ma, color="blue", alpha=0.2)
    ax.set_xlabel("x")
    ct_patch = mpatches.Patch(lw=1, linestyle="-", color="red", label="converge time")
    reward_patch = mpatches.Patch(lw=1, linestyle="-", color="blue", label="score")
    patch_set = [ct_patch, reward_patch]
    ax.legend(handles=patch_set)
    plt.show()
    fig.savefig("./test.png")


def plot(record):
    steps, mean, min, max = (record[i] for i in ["steps", "mean", "min", "max"])
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(steps, mean, color="blue", label="reward")
    ax.fill_between(steps, min, max, color="blue", alpha=0.2)
    ax.set_xlabel("number of steps")
    ax.set_ylabel("Average score per episode")
    reward_patch = mpatches.Patch(lw=1, linestyle="-", color="blue", label="score")
    patch_set = [reward_patch]
    ax.legend(handles=patch_set)
    fig.savefig("./performance.png")
