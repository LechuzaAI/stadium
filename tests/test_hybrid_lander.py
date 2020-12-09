from stadium.envs import HybridLander

NUM_EPISODES = 5

if __name__ == '__main__':
    env = HybridLander()
    s = env.reset()

    total_reward = 0.0
    done = False

    for episode in range(NUM_EPISODES):
        while not done:
            a = env.action_space.sample()

            state, reward, done, info = env.step(a)
            total_reward += reward

            env.render()

        print('Episode:\t{}\tTotal Reward:\t{}'.format(episode, total_reward))
        total_reward = 0.0
        env.reset()

    env.close()
