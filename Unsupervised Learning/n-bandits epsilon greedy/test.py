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


        self.cut_off = 0.2

        self.bandits = np.random.uniform(self.bandit_range_low, self.bandit_range_high,
                                            self.num_bandits)


        self.bandit_range = bandit_range_high - bandit_range_low
        self.bandit_idcs = range(self.num_bandits)

        # obtain pull function with specified paramters

        self.true_best_bandit = np.argmax(self.bandits)
        self.optimal_pay = self.bandits[self.true_best_bandit]/self.bandit_range
        self.true_scnd_best_bandit = np.argmax(np.delete(self.bandits,self.true_best_bandit))
        self.scnd_best_pay = self.bandits[self.true_scnd_best_bandit]/self.bandit_range


    def pull(self, bandit: float):
        p = random.uniform(self.bandit_range_low, self.bandit_range_high)
        if p <= bandit:
            return 1
        else:
            return 0



    def epsilon_greedy_pull(self, epsilon: float, best_bandit_idx: int,
                                all_but_best_bandit_idcs: Vec_int):
        p = random.uniform(0,1)
        # best_bandit_ = np.argmax(mean_pay)
        if p < epsilon:
            i = random.choice(all_but_best_bandit_idcs)
            return self.pull(self.bandits[i]), i
        else :
            return self.pull(self.bandits[best_bandit_idx]), best_bandit_idx




    def run_bandits(self):
        mean_pay = np.zeros(self.num_bandits) # mean observed pay per bandit initialization
        mean_total_pay = 0               # total observed pay initialization




        for i in self.bandit_idcs:
            pay = 0
            for _ in range(self.initial_sample_num):
                pay += self.pull(bandits[i])
            mean_pay[i] += pay/self.initial_sample_num

        # the overall pay
        mean_total_pay = np.mean(mean_pay)

        # number each bandit has been pulled
        times_pulled = self.initial_sample_num * np.ones(self.num_bandits)

        # number of total pulls
        N = self.initial_sample_num * self.num_bandits



        for i in range(self.num_episodes):
            current_best_bandit_idx = np.argmax(mean_pay)
            # current_best_bandit = bandits[idx_current_best_bandit]

            current_all_but_best_bandit_idcs = np.delete(self.bandit_idcs, np.argmax(mean_pay))

            # current_all_but_best_bandits = [bandits[i] for i in current_all_but_best_bandit_idcs]

            scnd_current_best_bandit_idx = np.argmax([mean_pay[i]
                                            for i in current_all_but_best_bandit_idcs])

            pay_best_bandit = mean_pay[current_best_bandit_idx]
            best_bandit_var = np.sqrt(pay_best_bandit*(1-pay_best_bandit)
                                        / times_pulled[current_best_bandit_idx])

            pay_scnd_best_bandit = mean_pay[scnd_current_best_bandit_idx]
            scnd_best_bandit_var = np.sqrt(pay_scnd_best_bandit*(1-pay_scnd_best_bandit)
                                        / times_pulled[scnd_current_best_bandit_idx])

            # epsilon = cut_off*(1-norm.ppf(max(pay_best_bandit - best_bandit_var,0),
            #                                 loc = pay_scnd_best_bandit,
            #                                 scale = scnd_best_bandit_var))

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
                episode_length *= 100
            else:
                episode_length *= int( 1 / np.log(1+epsilon))

            if N > 10000:
                return mean_pay, mean_total_pay, N
        return mean_pay, mean_total_pay, N


if __name__ == "__main__":

    num_bandits = 15

    bandit_range_low  = 0.
    bandit_range_high = 1.

    initial_sample_num = 15


    num_episodes = 10000

    num_tests = 1000

    tests_suceeded = 0

    pay_losses = np.zeros(num_tests)

    estim_scnd_best = np.zeros(num_tests)

    times_pulled = []

    for i in range(num_tests):
        Current_Bandits = Bandits(num_bandits, bandit_range_low, bandit_range_high,
                        initial_sample_num, num_episodes)

        bandits = Current_Bandits.bandits
        true_best_bandit = Current_Bandits.true_best_bandit
        optimal_pay = Current_Bandits.optimal_pay

        mean_pay, mean_total_pay, N = Current_Bandits.run_bandits()

        current_best = np.argmax(mean_pay)

        if np.argmax(mean_pay) == true_best_bandit:
            tests_suceeded += 1

        pay_losses[i] = (optimal_pay - mean_total_pay)/optimal_pay

        times_pulled.append(N)


    # if np.argmax(mean_pay) == true_best_bandit:
    #     print("Experiment succeeded, true best bandit detected")
    # else:
    #     print("Experiment failed, true best bandit not detected")

    print("Number of tests: {num_tests}".format(num_tests = num_tests))
    print("Number of bandits: {num_bandits}".format(num_bandits = num_bandits))
    print("Initial uniform sampling number: {initial_sample_num}".format(initial_sample_num = initial_sample_num))
    print("Number of episodes: {num_episodes}".format(num_episodes = num_episodes))
    print("Tests succeeded: {ratio:.2f}%".format(ratio = 100* tests_suceeded/num_tests))
    print("Average number of pulls: {M}".format(M = np.mean(times_pulled)))
    print("The maximum pay loss was : {max_pay_loss}".format(max_pay_loss = np.max(pay_losses)))
    print("The mean pay loss was : {mean_pay_loss}".format(mean_pay_loss = np.mean(pay_losses)))
    print("The std of pay loss was : {std_pay_loss}".format(std_pay_loss = np.std(pay_losses)))
