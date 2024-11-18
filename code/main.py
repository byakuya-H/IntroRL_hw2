#!/bin/env python
import logging, time
from tqdm import tqdm

from train import train, get_score
from algo import Make_Env, QAgent
from utils import plot, plot_fig, table
from arguments import get_conf


def main(conf: table):
    logging.info(
        f"time: {time.strftime('%D[%H:%M:%S]', time.localtime(time.time()))}, config: {dict(conf)}"
    )
    # agent, environment initial
    env = Make_Env(env_mode=conf.env_mode)
    agent = QAgent(
        env=env,
        lr=conf.lr,
        discount_factor=conf.discount_factor,
        default_action=conf.default_action,
    )

    # start train your agent
    records = train(
        conf.epochs,
        env,
        agent,
        conf.log_interval,
        render_epoch=conf.render_epoch,
        val_save_dir=conf.val_save_dir,
        conf=conf,
    )
    plot(records)
    logging.info("\n\n\n")


def test_lr(conf: table):
    logging.info("test learning rate")
    x, res = [], [[], [], [], []]
    add2res = lambda x: list(res[i].append(x[i]) for i in range(4))
    conf.update(render_epoch=0, log_interval=5, val_save_dir=None)
    for lr in tqdm(range(1, 100)):
        lr /= 100
        conf.update(lr=lr)
        x.append(lr)
        add2res(get_score(conf))
    plot_fig(x, *res)
    logging.info("\n\n\n")


# fmt: off
conf = get_conf(table(
    T=100,
    val_T=500,
    # val_T=50,
    epsilon=0.2,
    log_interval=40,
    epochs=1000,
    # epochs=10,
    # env_mode=2,
    lr=0.8,
    # discount_factor=0.8,
    # default_action=0,
    render_epoch=-1,
    val_save_dir="./imgs",
))
# fmt: on

# log_level = logging.DEBUG
log_level = logging.INFO
logging.basicConfig(level=log_level, filename="./log", filemode="a+")


if __name__ == "__main__":
    # main(conf)
    test_lr(conf)
