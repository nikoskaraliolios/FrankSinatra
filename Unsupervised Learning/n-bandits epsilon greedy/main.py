import numpy as np

from scipy.stats import norm

import random
from typing import List
Vec_int = List[int]
Vec = List[float]



class Bandits():


    def __init__(self, num_bandits :int, bandit_range_low: float,
                    bandit_range_high: float, initial_sample_num: int,
                    num_episodes: int):

        self.num_bandits = num_bandits
        self.bandit_range_low = bandit_range_low
        self.bandit_range_high = bandit_range_high
        self.initial_sample_num = initial_sample_num
        self.num_episodes = num_episodes


        self.cut_off = 0.25

        self.bandits = np.random.uniform(self.bandit_range_low, self.bandit_range_high,
                                            self.num_bandits)


        self.bandit_range = bandit_range_high - bandit_range_low
        self.bandit_idcs = range(self.num_bandits)

        # obtain pull function with specified paramters

        self.true_best_bandit = np.argmax(self.bandits)
        self.optimal_pay = self.bandits[self.true_best_bandit]/self.bandit_range


    def pull(self, bandit: float):
        p = random.uniform(self.bandit_range_low, self.bandit_range_high)
        if p <= bandit:
            return 1
        else:
            return 0


    def epsilon_greedy_pull(self, epsilon: float, best_bandit_idx: int,
                                all_but_best_bandit_idcs: Vec_int):
        p = random.uniform(0,1)
        if p < epsilon:
            i = random.choice(all_but_best_bandit_idcs)
            return self.pull(self.bandits[i]), i
        else :
            return self.pull(self.bandits[best_bandit_idx]), best_bandit_idx




    def run_bandits(self):
        mean_pay = np.zeros(self.num_bandits)   # mean observed pay per bandit initialization
        mean_total_pay = 0                      # total observed pay initialization


        # initial sampling, uniformly for all bandits

        for i in self.bandit_idcs:
            pay = 0
            for _ in range(self.initial_sample_num):
                pay += self.pull(bandits[i])
            mean_pay[i] += pay/self.initial_sample_num


        mean_total_pay = np.mean(mean_pay)

        # number each bandit has been pulled
        times_pulled = self.initial_sample_num * np.ones(self.num_bandits)

        # number of total pulls
        N = self.initial_sample_num * self.num_bandits


        for i in range(self.num_episodes):
            current_best_bandit_idx = np.argmax(mean_pay)

            current_all_but_best_bandit_idcs = np.delete(self.bandit_idcs, np.argmax(mean_pay))

            scnd_current_best_bandit_idx = np.argmax([mean_pay[i]
                                            for i in current_all_but_best_bandit_idcs])

            # calculate mean and std for estimated pay of two best performing
            # bandits

            pay_best_bandit = mean_pay[current_best_bandit_idx]
            best_bandit_var = np.sqrt(pay_best_bandit*(1-pay_best_bandit)
                                        / times_pulled[current_best_bandit_idx])

            pay_scnd_best_bandit = mean_pay[scnd_current_best_bandit_idx]
            scnd_best_bandit_var = np.sqrt(pay_scnd_best_bandit*(1-pay_scnd_best_bandit)
                                        / times_pulled[scnd_current_best_bandit_idx])

            # set epsilon as a fraction of the probability that the second best
            # bandit be better than the best one minus one std

            # epsilon = cut_off*(1-norm.ppf(max(pay_best_bandit - best_bandit_var,0),
            #                                 loc = pay_scnd_best_bandit,
            #                                 scale = scnd_best_bandit_var))
            #


            # set epsilon as the probability that the second best bandit be
            # better than the best one minus a fraction of its std


            if scnd_best_bandit_var == 0:
                epsilon = 0.001
            else:
                epsilon = 1-norm.cdf(pay_best_bandit -
                                        self.cut_off* best_bandit_var,
                                            loc = pay_scnd_best_bandit, scale = scnd_best_bandit_var)


            current_times_pulled = np.zeros(self.num_bandits)
            current_times_won    = np.zeros(self.num_bandits)

            if epsilon < 0.01:
                episode_length = 100
            else:
                episode_length = int( 1 / np.log(1+epsilon))

            for _ in range(episode_length):
                outcome, bandit_pulled = self.epsilon_greedy_pull(epsilon, current_best_bandit_idx,
                                                                current_all_but_best_bandit_idcs)

                current_times_pulled[bandit_pulled] += 1
                current_times_won[bandit_pulled]    += outcome

            # current_mean_pay = np.zeros(num_bandits)

            updated_times_pulled = times_pulled + current_times_pulled

            mean_pay = (current_times_won + times_pulled * mean_pay)/updated_times_pulled
            times_pulled = updated_times_pulled

            updated_N = N + episode_length
            mean_total_pay = (sum(current_times_won) + N*mean_total_pay)/updated_N
            N = updated_N

            if epsilon < 0.01:
                episode_length *= 1000
            else:
                episode_length = int( episode_length / (1+epsilon))

            if N > 10000:
                return mean_pay, mean_total_pay, N
        return mean_pay, mean_total_pay, N


if __name__ == "__main__":

    num_bandits = 15

    bandit_range_low  = 0.
    bandit_range_high = 1.

    initial_sample_num = 10

    num_episodes = 100

    Bandits = Bandits(num_bandits, bandit_range_low, bandit_range_high,
                        initial_sample_num, num_episodes)

    bandits = Bandits.bandits
    true_best_bandit = Bandits.true_best_bandit
    optimal_pay = Bandits.optimal_pay

    mean_pay, mean_total_pay, N = Bandits.run_bandits()

    if np.argmax(mean_pay) == true_best_bandit:
        print("Experiment succeeded, true best bandit detected")
    else:
        print("Experiment failed, true best bandit not detected")
    print("The bandits are : {bandits}".format(bandits = bandits))

    print("The estimated mean pay for each bandit is : {mean_pay}".format(mean_pay = mean_pay))
    print("Number of pulls : {N}".format(N = N))
    print("The mean total pay for this run is : {mean_total_pay:.2f}".format(mean_total_pay = mean_total_pay))
    print("The optimal pay for this game is : {optimal_pay:.2f}".format(optimal_pay = optimal_pay))
    print("The pay loss for this game is : {pay_loss:.2f}%".format(pay_loss = (optimal_pay - mean_total_pay)/optimal_pay))
