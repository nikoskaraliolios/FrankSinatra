## The epsilon greedy n-bandit problem

A first round of uniform sampling is performed, and then epsilon is chosen
as the probability that the second best performing bandit be better than
the first minus a fraction of the estimated standard deviation for the
pay for the best bandit.

The epsilon is updated increasingly rarely as it converges to 0.

A test is considered a success if the algorithm detected the best bandit.
The number of pulls in the tests was hardcoded to not exceed 10k by too much.
