"""
The same options as for pong, but for breakout
"""
import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame

import tensorflow as tf


summary_writer = tf.summary.FileWriter("logs/3-breakout-no-prio-replay-no-dueling")


def callback(lcl, glb):
    global summary_writer

    step = lcl['t']
    if step > 100:
        mean_reward = sum(lcl['episode_rewards'][-101:-1]) / 100.0
        if step % 1000 == 0:
            summary = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=mean_reward)])
            summary_writer.add_summary(summary, global_step=step)
            summary_writer.flush()
    return False


def main():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=False
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False,
        callback=callback
    )
    act.save("pong_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
