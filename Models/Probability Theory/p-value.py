from scipy.stats import binomtest

# Example data
# Number of successes and trials
successes = 20
trials = 25
# Hypothesized probability of success (e.g., 0.5 for a fair coin)
hypothesized_prob = 0.5

# Perform a binomial test
p_value = binomtest(successes, n=trials, p=hypothesized_prob)

print(f"P-value: {p_value}")