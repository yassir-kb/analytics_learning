# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt
#
# # set the random seed for reproducibility
# np.random.seed(123)
#
# # generate data for normal distribution
# normal_data = np.random.normal(loc=5, scale=2, size=1000)
#
# # calculate the mean and standard deviation of the data
# normal_mean = np.mean(normal_data)
# normal_std = np.std(normal_data)
#
# # generate data for exponential distribution
# exponential_data = np.random.exponential(scale=2, size=1000)
#
# # calculate the mean of the data
# exponential_mean = np.mean(exponential_data)
#
# # generate data for poisson distribution
# poisson_data = np.random.poisson(lam=3, size=1000)
#
# # calculate the mean of the data
# poisson_mean = np.mean(poisson_data)
#
# # generate data for gamma distribution
# gamma_data = np.random.gamma(shape=2, scale=2, size=1000)
#
# # calculate the mean and variance of the data
# gamma_mean = np.mean(gamma_data)
# gamma_var = np.var(gamma_data)
#
# # plot the normal distribution
# plt.hist(normal_data, density=True, alpha=0.5, label='data')
# x = np.linspace(normal_mean - 3*normal_std, normal_mean + 3*normal_std, 100)
# plt.plot(x, stats.norm.pdf(x, normal_mean, normal_std), label='normal distribution')
# plt.legend()
# plt.title('Normal Distribution')
# plt.show()
#
# # plot the exponential distribution
# plt.hist(exponential_data, density=True, alpha=0.5, label='data')
# x = np.linspace(0, 10, 100)
# plt.plot(x, stats.expon.pdf(x, scale=exponential_mean), label='exponential distribution')
# plt.legend()
# plt.title('Exponential Distribution')
# plt.show()
#
# # plot the poisson distribution
# plt.hist(poisson_data, density=True, alpha=0.5, label='data')
# x = np.arange(0, 15)
# plt.plot(x, stats.poisson.pmf(x, poisson_mean), 'bo', ms=8, label='poisson distribution')
# plt.legend()
# plt.title('Poisson Distribution')
# plt.show()
#
# # plot the gamma distribution
# plt.hist(gamma_data, density=True, alpha=0.5, label='data')
# x = np.linspace(0, 15, 100)
# plt.plot(x, stats.gamma.pdf(x, a=gamma_mean**2/gamma_var, scale=gamma_var/gamma_mean), label='gamma distribution')
# plt.legend()
# plt.title('Gamma Distribution')
# plt.show()


from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon, gamma, norm, poisson

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get user input
        distribution_type = request.form['distribution_type']
        num_samples = int(request.form['num_samples'])

        # Generate data
        if distribution_type == 'normal':
            data = norm.rvs(size=num_samples)
        elif distribution_type == 'uniform':
            data = np.random.uniform(size=num_samples)
        elif distribution_type == 'standard':
            data = np.random.randn(num_samples)
        elif distribution_type == 'exponential':
            data = expon.rvs(size=num_samples)
        elif distribution_type == 'poisson':
            data = poisson.rvs(mu=5, size=num_samples)
        elif distribution_type == 'gamma':
            data = gamma.rvs(a=5, size=num_samples)
        else:
            data = None

        # Generate plot
        if data is not None:
            plt.hist(data, bins=30)
            plt.title(f'{distribution_type.capitalize()} Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.savefig('static/plot.png')

            # Render template with plot
            return render_template('kolmogorov_smirnov.html', plot_url='static/plot.png')
        else:
            return render_template('kolmogorov_smirnov.html', error=True)
    else:
        return render_template('kolmogorov_smirnov.html')

if __name__ == '__main__':
    app.run(debug=True)
