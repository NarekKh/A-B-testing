from abc import ABC, abstractmethod
import logging
import random
import csv

# Set up logging
logging.basicConfig()
logger = logging.getLogger("MAB Application")
logging.basicConfig(level=logging.DEBUG)


class CustomFormatter(logging.Formatter):
    # Defining custom log formatting
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (line: %(lineno)d)"
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(CustomFormatter())


class Bandit(ABC):
    @abstractmethod
    def __init__(self, p):
        # Initializing the bandit with a probability distribution p
        self.p = p
        self.rewards = []  # Creating list to store rewards
        self.regrets = []  # Same, but with regrets

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    def experiment(self, num_trials):
        # Running an experiment with num_trials iterations
        for _ in range(num_trials):
            arm = self.pull()
            reward = Bandit_Reward[arm]
            self.rewards.append(reward)  # Recording the reward
            regret = max(Bandit_Reward) - reward  # Calculating regret
            self.regrets.append(regret)  # Recording regret
            self.update(arm, reward)  # Updating bandit's state

    def report(self, algorithm):
        # Reporting results of the experiment
        with open(f'{algorithm}_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bandit', 'Reward', 'Algorithm'])
            for i in range(len(self.rewards)):
                writer.writerow([i, self.rewards[i], algorithm])
        avg_reward = sum(self.rewards) / len(self.rewards)
        avg_regret = sum(self.regrets) / len(self.regrets)
        print(f'Average Reward for {algorithm}: {avg_reward}')
        print(f'Average Regret for {algorithm}: {avg_regret}')


class EpsilonGreedy(Bandit):
    def __init__(self, p, initial_epsilon):
        super().__init__(p)
        self.epsilon = initial_epsilon  # Initializing epsilon
        self.q_values = [0] * len(p)  # Initializing q-values
        self.action_counts = [0] * len(p)  # Initializing action counts

    def __repr__(self):
        return 'EpsilonGreedy'

    def pull(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.p) - 1)
        else:
            return self.q_values.index(max(self.q_values))

    def update(self, arm, reward):
        self.action_counts[arm] += 1  # Incrementing action count
        self.q_values[arm] += (reward - self.q_values[arm]) / self.action_counts[arm]  # Updating q-value
        self.epsilon = 1 / (sum(self.action_counts) + 1)  # Decay epsilon


class ThompsonSampling(Bandit):
    def __init__(self, p, precision):
        super().__init__(p)
        self.precision = precision  # Setting precision
        self.alpha = [1.0] * len(p)  # Initializing alpha values
        self.beta = [1.0] * len(p)  # Initializing beta values

    def __repr__(self):
        return 'ThompsonSampling'

    def pull(self):
        samples = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.p))]
        return samples.index(max(samples))

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        if self.alpha[arm] <= 0 or self.beta[arm] <= 0:
            self.alpha[arm] = 1.0  # Resetting alpha if it becomes non-positive
            self.beta[arm] = 1.0  # Resetting beta if it becomes non-positive
        logging.debug(f'ThompsonSampling - Arm {arm} selected, Reward: {reward}')


class Visualization:
    @staticmethod
    def plot1(epsilon_greedy_rewards, thompson_rewards):
        import matplotlib.pyplot as plt

        plt.plot(epsilon_greedy_rewards, label='Epsilon Greedy')
        plt.plot(thompson_rewards, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Reward')
        plt.title('Learning Process')
        plt.legend()
        plt.show()

    @staticmethod
    def plot2(e_greedy_rewards, thompson_rewards):
        import matplotlib.pyplot as plt

        cumulative_e_greedy_rewards = [sum(e_greedy_rewards[:i + 1]) for i in range(len(e_greedy_rewards))]
        cumulative_thompson_rewards = [sum(thompson_rewards[:i + 1]) for i in range(len(thompson_rewards))]

        plt.plot(cumulative_e_greedy_rewards, label='Epsilon Greedy')
        plt.plot(cumulative_thompson_rewards, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards')
        plt.legend()
        plt.show()


def comparison():
    pass

# Running
if __name__ == '__main__':
    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000

    epsilon_value = 0.1
    precision_value = 0.001

    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon_value)
    thompson_sampling_bandit = ThompsonSampling(Bandit_Reward, precision_value)

    epsilon_greedy_bandit.experiment(NumberOfTrials)
    thompson_sampling_bandit.experiment(NumberOfTrials)

    epsilon_greedy_bandit.report('EpsilonGreedy')
    thompson_sampling_bandit.report('ThompsonSampling')


eg_rewards = [2, 3, 1, 4, 2]
ts_rewards = [3, 2, 4, 1, 3]
eg_cumulative_rewards = [2, 1, 3, 2, 4]
ts_cumulative_rewards = [4, 3, 2, 1, 2]

visualization = Visualization()
visualization.plot1(eg_rewards, ts_rewards)
visualization.plot2(eg_cumulative_rewards, ts_cumulative_rewards)
