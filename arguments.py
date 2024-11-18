import argparse

from utils import table, merge_conf

# fmt: off
default_args = [
    ["num_stacks", dict(type=int, default=4)],
    ["T", dict(type=int, default=100)],
    ["val_T", dict(type=int, default=2000)],
    ["epochs", dict(type=int, default=1000)],
    ["env_mode", dict(type=int, default=2)],
    ## other parameter
    ["log_interval", dict(type=int, default=10, help="log interval, one log per n updates (default: 10)")],
    # ["save_img", dict(type=bool, default=True)],
    # ["save_interval", dict(type=int, default=10, help="save interval, one eval per n updates (default: None)")],
    ["lr", dict(type=float, default=1.0, help="learning rate `\\alpha` in q-learning")],
    ["discount_factor", dict(type=float, default=0.8, help="discount factor `\\gamma` in q-learning")],
    ["default_action", dict(type=int, default=0, help="initial action for policy")],
    ["render_epoch", dict(type=int, default=0, help="whether render visual feedback, 0 for None, 1 for all val epochs, -1 for the last validation.")],
    ["val_save_dir", dict(type=str, default="", help="save result in dir.")],
]
# fmt: on


def get_args():
    global default_args
    parser = argparse.ArgumentParser(description="RL")
    for arg in default_args:
        parser.add_argument("--" + arg[0].replace("_", "-"), **arg[1])
    return parser.parse_args()


def get_conf(conf: table):
    global default_args, get_args
    args = get_args()
    conf = merge_conf(conf, {arg[0]: arg[1]["default"] for arg in default_args})
    conf = merge_conf(
        conf,
        {
            arg[0]: getattr(args, arg[0])
            for arg in default_args
            if arg[1]["default"] != getattr(args, arg[0])
            or (
                isinstance(arg[1]["default"], float)
                and abs(arg[1]["default"] - getattr(args, arg[0])) > 1e-6
            )
        },
    )
    return conf
