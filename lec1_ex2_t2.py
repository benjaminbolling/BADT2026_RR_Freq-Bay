import numpy as np
from scipy.stats import beta, norm
from math import sqrt, log, exp

# Data - PoC (Scales et al (2003))
y0, n0 = 4, 15 # no masks
y1, n1 = 3, 16 # with masks

# # Data 1 (Hall et al (2014))
# y0, n0 = 0, 6 # no masks
# y1, n1 = 0, 42 # with masks


conf = 0.90 # 95% CI

print('\n\ny0 = {}, n0 = {}'.format(y0,n0))
print('y1 = {}, n1 = {}\n'.format(y1,n1))


# # # Frequentist solution # # #

if y0 == 0 or y1 == 0: # add 0.5 as continuity
    p0_hat = (y0 + 0.5) / (n0 + 1)
    p1_hat = (y1 + 0.5) / (n1 + 1)
else:
    p0_hat = y0 / n0
    p1_hat = y1 / n1

freq_RR_hat = p1_hat / p0_hat

z = norm.ppf(1 - (1 - conf) / 2)

if y0 == 0 or y1 == 0: # add 0.5 as continuity
    se_log_rr = sqrt(
    (1 / (y1 + 0.5)) - (1 / (n1 + 1)) +
    (1 / (y0 + 0.5)) - (1 / (n0 + 1))
    )
else:
    se_log_rr = sqrt(
        (1 / y1) - (1 / n1) +
        (1 / y0) - (1 / n0)
    )

log_rr = log(freq_RR_hat)

ci_lower = exp(log_rr - z * se_log_rr)
ci_upper = exp(log_rr + z * se_log_rr)

if y0 == 0 or y1 == 0:
    print('Warning: Added 0.5 as continuity due to 0-division for frequentist RR')
print('Frequentist plug-in RR ({}% CI): {:.2f} ({:.2f}-{:.2f})'.format(conf*100, freq_RR_hat, ci_lower, ci_upper))

# # # Bayes # # #

# Posterior parameters
a0, b0 = y0 + 1, n0 - y0 + 1
a1, b1 = y1 + 1, n1 - y1 + 1

# Draw posterior samples
N = 100_000
pi0_samples = beta.rvs(a0, b0, size=N)
pi1_samples = beta.rvs(a1, b1, size=N)

# Relative risk samples
Bayes_RR_samples = pi1_samples / pi0_samples

Bayes_RR_mean = np.mean(Bayes_RR_samples)

Bayes_CI_lvl = [
    100 * (1 - conf) / 2, 
    100 * (1 - (1 - conf) / 2)
    ]

Bayes_RR_ci = np.percentile(Bayes_RR_samples, [2.5, 97.5])

print('Bayesian posterior mean RR ({}% CI): {:.2f} ({:.2f}-{:.2f})\n'.format(conf*100, Bayes_RR_mean, Bayes_RR_ci[0], Bayes_RR_ci[1]))
