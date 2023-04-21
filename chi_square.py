# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
#
# # Generate data for normal distribution
# mu, sigma = 0, 1
# n = 1000
# data_norm = np.random.normal(mu, sigma, n)
#
# # Generate data for exponential distribution
# beta = 1
# data_exp = np.random.exponential(beta, n)
#
# # Generate data for Poisson distribution
# lam = 5
# data_poisson = np.random.poisson(lam, n)
#
# # Generate data for gamma distribution
# shape, scale = 2, 2
# data_gamma = np.random.gamma(shape, scale, n)
#
# # Define bin edges and expected frequencies
# bins = 20
# norm_bins = np.linspace(-4, 4, bins+1)
# exp_bins = np.linspace(0, 10, bins+1)
# poisson_bins = np.arange(0, 20, 1)
# gamma_bins = np.linspace(0, 20, bins+1)
#
# norm_expected = n*np.diff(stats.norm.cdf(norm_bins, mu, sigma))
# exp_expected = n*np.diff(stats.expon.cdf(exp_bins, 0, beta))
# poisson_expected = n*np.diff(stats.poisson.cdf(poisson_bins, lam))
# gamma_expected = n*np.diff(stats.gamma.cdf(gamma_bins, shape, scale))
#
# # Perform chi-square goodness-of-fit test and plot results for each distribution
# fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#
# # Normal distribution
# norm_observed, norm_bins, _ = axs[0, 0].hist(data_norm, bins=norm_bins, density=True, alpha=0.5)
# norm_chi2, norm_p = stats.chisquare(norm_observed, norm_expected)
# axs[0, 0].set_title(f"Normal Distribution: p-value = {norm_p:.4f}")
# axs[0, 0].plot(norm_bins, stats.norm.pdf(norm_bins, mu, sigma)*n/bins, lw=2)
#
# # Exponential distribution
# exp_observed, exp_bins, _ = axs[0, 1].hist(data_exp, bins=exp_bins, density=True, alpha=0.5)
# exp_chi2, exp_p = stats.chisquare(exp_observed, exp_expected)
# axs[0, 1].set_title(f"Exponential Distribution: p-value = {exp_p:.4f}")
# axs[0, 1].plot(exp_bins, stats.expon.pdf(exp_bins, 0, beta)*n/bins, lw=2)
#
# # Poisson distribution
# poisson_observed, poisson_bins, _ = axs[1, 0].hist(data_poisson, bins=poisson_bins, density=True, alpha=0.5)
# poisson_chi2, poisson_p = stats.chisquare(poisson_observed, poisson_expected)
# axs[1, 0].set_title(f"Poisson Distribution: p-value = {poisson_p:.4f}")
# axs[1, 0].plot(poisson_bins, stats.poisson.pmf(poisson_bins, lam)*n/bins, lw=2)
#
# # Gamma distribution
# gamma_observed, gamma_bins, _ = axs[1, 1].hist(data_gamma, bins=gamma_bins, density=True, alpha=0.5)
# gamma_chi2, gamma_p = stats.chisquare(gamma_observed, gamma_expected)
# axs[1, 1].set_title


from flask import Flask, render_template, request
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get distribution type from form submission
        distribution_type = request.form['distribution_type']

        # Generate random sample data for selected distribution
        if distribution_type == 'normal':
            data = np.random.normal(loc=0, scale=1, size=100)
        elif distribution_type == 'exponential':
            data = np.random.exponential(scale=1, size=100)
        elif distribution_type == 'poisson':
            data = np.random.poisson(lam=1, size=100)
        elif distribution_type == 'gamma':
            data = np.random.gamma(shape=2, scale=1, size=100)

        # Calculate expected values for selected distribution
        if distribution_type == 'normal':
            expected = stats.norm(loc=np.mean(data), scale=np.std(data)).pdf(data)
        elif distribution_type == 'exponential':
            expected = stats.expon(scale=np.mean(data)).pdf(data)
        elif distribution_type == 'poisson':
            expected = stats.poisson(mu=np.mean(data)).pmf(data)
        elif distribution_type == 'gamma':
            expected = stats.gamma(a=np.mean(data)**2/np.var(data), scale=np.var(data)/np.mean(data)).pdf(data)

        # Calculate Chi-square statistic and p-value
        chisq_stat, p_value = stats.chisquare(f_obs=np.histogram(data, bins='auto')[0], f_exp=expected)

        # Create histogram of sample data and plot expected distribution
        plt.hist(data, bins='auto', density=True, alpha=0.7)
        plt.plot(data, expected, 'r-', lw=2)
        plt.title(f'{distribution_type.capitalize()} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.savefig('static/plot.png')

        # Render template with plot image and results
        return render_template('chi_square.html', plot_path='static/plot.png', chisq_stat=chisq_stat, p_value=p_value)

    else:
        # Render initial form
        return render_template('chi_square.html')

if __name__ == '__main__':
    app.run(debug=True)
