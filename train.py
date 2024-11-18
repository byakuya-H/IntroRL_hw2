from typing import List, Tuple, Union
import time, logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import table, plot, Drawer, save_result
from env import Make_Env
from algo import Agent, QAgent


def q_learning(
    epoch: int,
    env: Make_Env,
    agent: QAgent,
    T: int,
    epsilon: float,
):
    logging.debug(f"start to train on epoch: {epoch}")
    obs = tuple(env.reset().astype(int))
    render = epoch % 100 == 0 and logging.getLogger().isEnabledFor(logging.DEBUG)
    draw = Drawer(plt).draw if render else lambda _: None
    # for _ in range(T):
    for _ in tqdm(range(T), leave=False):
        action = agent.select_action(obs, epsilon=epsilon)
        obs_next, reward, done, info = env.step(action)
        draw(info)
        obs_next = tuple(obs_next.astype(int))
        agent.eval_pol(obs, obs_next, action, reward)
        agent.update_pol(state=obs, act=action)
        logging.debug(f"obs:{obs} obs_next:{obs_next} action:{action} reward:{reward}")
        # if the episode has terminated, we need to reset the environment.
        obs = tuple(env.reset().astype(int)) if done else obs_next
    logging.debug(f"epoch {epoch} training ends")


def val(
    epoch: int,
    env: Make_Env,
    agent: Agent,
    T: int,
    start: float,
    render: bool = False,
    get_imgs: bool = False,
) -> Tuple[int, float, float, float, List[np.ndarray]]:
    logging.info(f"start to validate the agent: {agent} on epoch {epoch}")
    total_steps, rewards, total_reward, obs, infos = (
        (epoch + 1) * T,
        [],
        0,
        tuple(env.reset()),
        [],
    )
    draw = Drawer(plt).draw if render else lambda _: None
    # for _ in range(T):
    for _ in tqdm(range(T), leave=False):
        action = agent.select_action(obs)
        obs_next, reward, done, info = env.step(action)
        obs_next = tuple(obs_next)
        draw(info)
        if get_imgs:
            infos.append(info)
        total_reward += reward
        logging.debug(f"obs:{obs} obs_next:{obs_next} action:{action} reward:{reward}")
        if done:
            rewards.append(total_reward)
            total_reward, obs = 0, tuple(env.reset())
        else:
            obs = obs_next

    end = time.time()
    rwd_mean, rwd_max, rwd_min = (
        np.mean(rewards) if len(rewards) != 0 else 0,
        np.max(rewards) if len(rewards) != 0 else 0,
        np.min(rewards) if len(rewards) != 0 else 0,
    )
    logging.info(
        f"TIME {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))} "
        + f"Updates {epoch}, num timesteps {total_steps}, FPS {int(total_steps / (end - start))} \n "
        + f"avrage/min/max reward {rwd_mean:.1f}/{rwd_min:.1f}/{rwd_max:.1f}"
    )

    return (
        total_steps,
        rwd_mean,
        rwd_min,
        rwd_max,
        infos,
    )


def train(
    epochs: int,
    env: Make_Env,
    agent: QAgent,
    log_interval: int,
    conf: table,
    render_epoch: int = 0,
    val_save_dir: Union[str, None] = None,
) -> table:
    record, start = table(steps=[0], mean=[0], min=[0], max=[0]), time.time()
    add2rec = lambda res: list(record[k].append(v) for k, v in zip(record.keys(), res))
    # for epoch in range(epochs):
    for epoch in tqdm(range(epochs)):
        q_learning(epoch, env, agent, conf.T, conf.epsilon)
        if (epoch + 1) % log_interval == 0:
            res = val(
                epoch,
                Make_Env(),
                agent,
                conf.val_T,
                start,
                render=render_epoch > 0,
            )[:-1]
            add2rec(res)
    res = val(
        epochs,
        Make_Env(),
        agent,
        conf.val_T,
        start,
        render=(render_epoch == -1),
        get_imgs=True,
    )
    add2rec(res[:-1])
    if val_save_dir is not None and val_save_dir != "":
        save_result(val_save_dir, res[-1], agent.Qmodel())
    return record


def get_score(conf: table):
    logging.info(
        f"test the config: {dict(conf)}, time: {time.strftime('%D[%H:%M:%S]', time.localtime(time.time()))}"
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
    s, e, gap = 0, len(records.steps) - 1, max(records.mean) - min(records.min)
    while s < e and e - s >= 10:
        m = (s + e) // 2
        if np.var(records.mean[m:e]) ** 0.5 / gap < 7e-2:
            e = m
        else:
            s = m + 1
    logging.info("---***---\n\n\n")
    return (records.steps[e],) + tuple(
        np.mean(records[i][e:]) for i in ["mean", "min", "max"]
    )
