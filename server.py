#!/usr/bin/env python

from flask import Flask, render_template, request
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define routes
@app.route('/chi_square', methods=['GET', 'POST'])
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

@app.route('/kolmogorov_smirnov', methods=['GET', 'POST'])
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
