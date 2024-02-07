import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants
import math


from scipy.stats import rayleigh
import scipy.stats as stats
from scipy.special import erfinv
from scipy.stats import beta
from scipy.stats import gaussian_kde
from scipy.stats import chi2
from scipy.stats import rice
import scipy.special as sp

 ####### Rayleigh Distribution ##########
def Rayleigh_CDF(n,sigma):
    prob = np.random.uniform(0,1,n)
    Y_CDF = 1 - np.exp(-prob**2 / (2 * sigma**2))
    return Y_CDF

def interves_Rayleigh_CDF(n,sigma):
    prob = np.random.uniform(0,1,n)
    X_RV = np.sqrt(-2 * sigma**2 * np.log(1 - prob))

    return X_RV

def Emperical_Rayleigh_Probability(x, scale_parameter):
    y_emperical = (x / (scale_parameter**2)) * np.exp(-x**2 / (2 * scale_parameter**2))
    return y_emperical

def Rayleigh_Probability(x, scale_parameter):

    y = rayleigh.pdf(x, scale_parameter)
    return y

# Compute the CDF values
# n  = 10000
# sigma = 1


def Rayleigh_Pdf_Comparation(n,sigma):
    # Input data
    Rayleigh_data_pdf = interves_Rayleigh_CDF(n,sigma)

    scale_parameter = np.std(Rayleigh_data_pdf)
    Rayleigh_Pdf_x = np.linspace(0, max(Rayleigh_data_pdf), n)
        
    Rayleigh_Pdf_y1 = Emperical_Rayleigh_Probability(Rayleigh_Pdf_x, sigma)
    Rayleigh_Pdf_y2 = Rayleigh_Probability(Rayleigh_Pdf_x, sigma)

    return Rayleigh_data_pdf,Rayleigh_Pdf_x, Rayleigh_Pdf_y1,Rayleigh_Pdf_y2


def Rayleigh_Cdf_comparation(n,sigma):
    Rayleigh_data_Cdf = interves_Rayleigh_CDF(n,sigma)
    Rayleigh_data_Cdf.sort()

    #compute the cdf of RV
    rayleigh_RV_values = np.arange(1, len(Rayleigh_data_Cdf) + 1) / len(Rayleigh_data_Cdf)

    # Compute the Rayleigh CDF
    scale_parameter = np.std(Rayleigh_data_Cdf)
    rayleigh_cdf = rayleigh.cdf(Rayleigh_data_Cdf, scale=scale_parameter)

    return Rayleigh_data_Cdf, rayleigh_RV_values, rayleigh_cdf

#### Lognormal(mu, sigma^2) ###

def inverse_Lognormal_CDF(n,u,sigma):
  
    # Generate random uniform data between 0 and 1
    uniform_data = np.random.rand(n)

    # Use the inverse transform method with erfinv from scipy.special
    inv_logN_RV = np.exp(u + sigma * np.sqrt(2) * erfinv(2 * uniform_data - 1))

    return inv_logN_RV

def inverse_Lognormal_CDF_Scipy(n,u,sigma):
    # Define the parameters for the inverse lognormal distribution
    u = u  
    num_samples = n  

    # Create a lognormal distribution object with the given parameters
    lognormal_dist = stats.lognorm(s=sigma, scale=np.exp(u))

    # Generate random uniform data between 0 and 1
    uniform_data = np.random.rand(num_samples)

    # Use the inverse CDF of the lognormal distribution to get inverse lognormal data
    inv_logN_RV = lognormal_dist.ppf(uniform_data)
    return inv_logN_RV


def Emperical_Lognormal_PDF (x, u, scale_parameter):

    y_emperical = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - u) ** 2 / (2 * sigma ** 2))

    return y_emperical

def Rayleigh_Lognormal_PDF_scipy(x, u, scale_parameter):

    # Create a lognormal distribution object with the given parameters
    lognormal_dist = stats.lognorm(sigma, np.exp(u))

    # Compute the PDF for your data
    y = lognormal_dist.pdf(x)

    return y

# n  = 10000
# sigma = 1
# u = 0

def Lognormal_Pdf_Comparation(n,u, sigma):
    # Input data
    Lognormal_Pdf_data = inverse_Lognormal_CDF(n,u,sigma)

    scale_parameter = np.std(Lognormal_Pdf_data)
    Lognormal_Pdf_x = np.linspace(0, max(Lognormal_Pdf_data), n)
        
    Lognormal_Pdf_y1 = Emperical_Lognormal_PDF (Lognormal_Pdf_x, u, scale_parameter)
    Lognormal_Pdf_y2 = Rayleigh_Lognormal_PDF_scipy(Lognormal_Pdf_x, u, scale_parameter)

    return Lognormal_Pdf_data, Lognormal_Pdf_x, Lognormal_Pdf_y1,Lognormal_Pdf_y2



# n  = 10000
# sigma = 1
# u = 0

def Lognormal_Cdf_Comparation(n,u,sigma):
    # def Rayleigh_CDF_comparation(n,u,sigma):
    Lognormal_Cdf_data = inverse_Lognormal_CDF(n,u,sigma)
    Lognormal_Cdf_data.sort()

    #compute the cdf of RV
    Lognormal_Cdf_RV_values = np.arange(1, len(Lognormal_Cdf_data) + 1) / len(Lognormal_Cdf_data)

    # Compute the Rayleigh CDF

    lognormal_dist = stats.lognorm(sigma, np.exp(u))
    Lognormal_Cdf_values = lognormal_dist.cdf(Lognormal_Cdf_data)

    return Lognormal_Cdf_data, Lognormal_Cdf_RV_values, Lognormal_Cdf_values


#### Beta(alpha, beta) ####


def inverse_Beta_CDF(alpha, beta, n):
    
    # Generate random uniform data between 0 and 1
    uniform_data = np.random.rand(n)

    # Use the inverse transform method with erfinv from scipy.special
    x = (1 - u) ** (1 / alpha)
    inv_beta_RV = (1 - uniform_data) ** (1 / beta)

    
    return inv_beta_RV

def inverse_Beta_CDF_Scipy(alpha, beta, n):
    u = np.random.rand(n)  # Random value between 0 and 1
    inv_beta_RV = stats.beta.ppf(u, alpha, beta)

    return inv_beta_RV


def Emperical_Beta_PDF (x, alpha, beta):

    # Calculate the PDF using the estimated parameters
    y_emperical = (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))

    # Normalize the PDF
    y_emperical /= np.trapz(y_emperical, x)

    return y_emperical
    

def Beta_PDF_scipy(x, alpha, beta):

    # Compute the PDF for your data
    y = stats.beta.pdf(x, alpha, beta)

    return y

def Beta_Pdf_Comparation(n, alpha, beta):
    # Input data
    Beta_Pdf_data = inverse_Beta_CDF_Scipy(alpha, beta, n)

    scale_parameter = np.std(Beta_Pdf_data)
    Beta_Pdf_x = np.linspace(0, max(Beta_Pdf_data), n)
        
    Beta_Pdf_y1 = Emperical_Beta_PDF (Beta_Pdf_x, alpha, beta)
    Beta_Pdf_y2 = Beta_PDF_scipy(Beta_Pdf_x, alpha, beta)

    return Beta_Pdf_data, Beta_Pdf_x, Beta_Pdf_y1, Beta_Pdf_y2


def Beta_Cdf_Comparation(n,alpha, beta):
    # def Rayleigh_CDF_comparation(n,u,sigma):
    Beta_Cdf_data = inverse_Beta_CDF_Scipy(alpha, beta, n)
    Beta_Cdf_data.sort()

    #compute the cdf of RV
    Beta_Cdf_RV_values = np.arange(1, len(Beta_Cdf_data) + 1) / len(Beta_Cdf_data)

    # Compute the Rayleigh CDF
 

    # Create a range of x values for the CDF
    Beta_Cdf_x = np.linspace(0, max(Beta_Cdf_data), n)

    # Calculate the CDF using the estimated parameters
    beta_cdf = stats.beta.cdf(Beta_Cdf_x, alpha, beta)

    return Beta_Cdf_data, Beta_Cdf_RV_values, Beta_Cdf_x, beta_cdf


#### chi_square(alpha, beta) #### 

def inverse_chi_square_CDF(k,n):

    # Generate a random number, u, from a uniform distribution in the range [0, 1]
    u = np.random.rand(n)

    # Calculate the quantile, x, using the inverse CDF of the chi-square distribution
    inverse_chi_squared_data = (-2 * np.log(1 - u)) * k
            
    return inverse_chi_squared_data

def inverse_chi_square_CDF_Scipy(k,n):

    # Generate a random number from the inverse CDF
    u = np.random.rand(n)  # Random value between 0 and 1
    inv_chi_squared_RV = stats.chi2.ppf(u, k)


    return inv_chi_squared_RV


def Emperical_chi_square_PDF (x, k):

    y_emperical = (1 / (2**(k/2) * np.math.gamma(k/2)) * x**(k/2 - 1) * np.exp(-x/2))

    # Normalize the PDF
    y_emperical /= np.trapz(y_emperical, x)

    return y_emperical
    

def chi_square_PDF_scipy(x, k):

    y = stats.chi2.pdf(x, k)

    return y


def chi_square_Pdf_Comparation(k,n):
    # Input data
    chi_square_Pdf_data = inverse_chi_square_CDF_Scipy(k,n)

    scale_parameter = np.std(chi_square_Pdf_data)
    chi_square_Pdf_x = np.linspace(0, max(chi_square_Pdf_data), n)
        
    chi_square_Pdf_y1 = Emperical_chi_square_PDF (chi_square_Pdf_x, k)
    chi_square_Pdf_y2 = chi_square_PDF_scipy(chi_square_Pdf_x, k)

    return chi_square_Pdf_data, chi_square_Pdf_x, chi_square_Pdf_y1, chi_square_Pdf_y2

def chi_square_Cdf_Comparation(n, k):
    # def Rayleigh_CDF_comparation(n,u,sigma):
    chi_square_Cdf_data = inverse_chi_square_CDF_Scipy(k,n)
    chi_square_Cdf_data.sort()

    #compute the cdf of RV
    # chi_square_Cdf_RV_values = np.arange(1, len(chi_square_Cdf_data) + 1) / len(chi_square_Cdf_data)
    chi_square_Cdf_RV_values = stats.chi2.cdf(chi_square_Cdf_data , k)

    # Compute the Rayleigh CDF
    chi_square_Cdf_x = np.linspace(0, max(chi_square_Cdf_data), n)

    chi_square_Cdf = stats.chi2.cdf(chi_square_Cdf_x , k) 

    return chi_square_Cdf_x, chi_square_Cdf_RV_values, chi_square_Cdf 

 #### Rice distribution (nu,sigma) #### 


def inverse_Rice_CDF(n, u, sigma):
    
    # Generate a random number from the inverse Rice distribution
    # u is the Degrees of freedom
    uniform = np.random.rand(n)  
    inverse_Rice_data = sigma * np.sqrt(-2 * np.log(uniform)) + u
            
    return inverse_Rice_data

def inverse_Rice_CDF_Scipy(n, u, sigma):

    uniform = np.random.rand(n) 
    inverse_Rice_RV =  stats.rice.ppf(uniform, u / sigma, scale=sigma)

    return inverse_Rice_RV
    

def Rice_PDF_scipy(x,n):

    # Calculate the PDF for each x
    uniform = np.random.rand(n) 
    y = stats.rice.pdf(x, u/sigma, scale=sigma)

    return y

def Rice_Pdf_Comparation(n, u, sigma):
    # Input data
    Rice_Pdf_data = inverse_Rice_CDF_Scipy(n, u, sigma)

        # Estimate parameters (ν and σ) from your data
    v_estimate = np.sqrt(np.mean(Rice_Pdf_data ** 2))
    sigma = np.sqrt(np.mean((Rice_Pdf_data - v_estimate) ** 2))

    
    Rice_Pdf_x = np.linspace(0, max(Rice_Pdf_data), n)
        
    Rice_Pdf_y2 = Rice_PDF_scipy(Rice_Pdf_x,n)

    return Rice_Pdf_data, Rice_Pdf_x, Rice_Pdf_y2

def Rice_Cdf_Comparation(n, u, sigma):
    # Generate random data from a Rice distribution
    data = inverse_Rice_CDF_Scipy(n, u, sigma)  # You should have a method for this

    # Sort the data
    Rice_Cdf_data = sorted(data)

    # Compute the sample mean and sample standard deviation
    sample_mean = np.mean(Rice_Cdf_data)
    sample_stddev = np.std(Rice_Cdf_data, ddof=1)

    # Estimate the scale parameter (b) using MLE
    b_estimated = np.sqrt(sample_mean**2 + (sample_stddev**2) / 2)

    # Calculate the CDF of the Rice distribution based on the estimated b
    
    Rice_Cdf_RV_values = stats.rice.cdf(Rice_Cdf_data, b_estimated)

    # Create a range of x values for CDF calculation
    Rice_Cdf_x = np.linspace(0, max(Rice_Cdf_data), n)

    # Calculate the CDF for each x
    Rice_Cdf = stats.rice.cdf(Rice_Cdf_x, b_estimated)

    return Rice_Cdf_data, Rice_Cdf_x, Rice_Cdf_RV_values, Rice_Cdf



#-------------------------------------Simulation Loop-----------------------------------------#
# function_list = ['Rayleigh','Lognormal','Beta','Chi-square','Rice']
# Experiments_number = [1000,10000,100000]

function_list = ['Rayleigh', 'Lognormal', 'Beta', 'Chi-square', 'Rice']
Experiments_number = [1000, 10000, 100000]


for i, string in enumerate(function_list):
    # for string in function_list:
    if string == 'Rayleigh':
        
        # Create a figure and subplots
        num_cols = len(Experiments_number)
        fig, axes = plt.subplots(num_cols,2, figsize=(12,12))

        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=3, ncols=1)
       

        for j, (n,subfig) in enumerate(zip(Experiments_number,subfigs)):
            
            subfig.subplots_adjust(top=0.85)
            subfig.suptitle(f'For the distribution of { string}, The number of samples is{ n}', fontsize=16, color = "navy", fontweight="bold")

            # print(f'---------------------------------------------------------')
            # print(f'For the distribution of { string}, The number of samples is{ n}')

            sigma = 1
            Rayleigh_data_pdf,Rayleigh_Pdf_x, Rayleigh_Pdf_y1,Rayleigh_Pdf_y2 =  Rayleigh_Pdf_Comparation(n,sigma)
            Rayleigh_data_Cdf, rayleigh_RV_values, rayleigh_cdf = Rayleigh_Cdf_comparation(n,sigma)


            # Plot data on each subplot
            subfig.suptitle(f'For the distribution of { string}, The number of samples is{ n}', fontsize=16)
            # fig[1].suptitle()
            axes[j][0].hist(Rayleigh_data_pdf, bins=100, density=True, alpha=0.6, color='b', label='Data Histogram')
            axes[j][0].plot(Rayleigh_Pdf_x, Rayleigh_Pdf_y1, 'g-', lw=2, label='Rayleigh PDF Fit')
            axes[j][0].set_xlabel('Random Numbers')
            # axes[j][0].set_xlim(0, n)
            axes[j][0].set_ylabel('Probability Density')
            axes[j][0].set_title('Rayleigh Distribution Fit to Input Data')
            axes[j][0].grid()
            axes[j][0].legend()

            axes[j][1].plot(Rayleigh_data_Cdf, rayleigh_RV_values, marker='*', linestyle='-', color='g', label='Empirical CDF')
            axes[j][1].plot(Rayleigh_data_Cdf, rayleigh_cdf, color='r', linestyle='--', label='Rayleigh CDF')
            axes[j][1].set_xlabel('RV data')
            # axes[j][1].set_xlim(0, n)
            axes[j][1].set_ylabel('Rayleigh_CDF')
            axes[j][1].set_title('Empirical CDF vs. Rayleigh CDF')
            axes[j][1].grid()
            axes[j][1].legend()

        # Adjust the layout
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=8.0)

        # Show the plot
        plt.show()

                

    elif string == 'Lognormal':
        # Create a figure and subplots
        num_cols = len(Experiments_number)
        fig, axes = plt.subplots(num_cols,2, figsize=(12,12))

        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=3, ncols=1)
       

        for j, (n,subfig) in enumerate(zip(Experiments_number,subfigs)):
            
            subfig.subplots_adjust(top=0.85)
            subfig.suptitle(f'For the distribution of { string}, The number of samples is{ n}', fontsize=16, color = "navy", fontweight="bold")
            
            sigma = 1
            u = 0
            Lognormal_Pdf_data, Lognormal_Pdf_x, Lognormal_Pdf_y1,Lognormal_Pdf_y2 = Lognormal_Pdf_Comparation(n,u,sigma)
            Lognormal_Cdf_data, Lognormal_Cdf_RV_values, Lognormal_Cdf_values = Lognormal_Cdf_Comparation(n,u,sigma)


            # Plot data on each subplot
            axes[j][0].hist(Lognormal_Pdf_data, bins=100, density=True, alpha=0.6, color='b', label='Data Histogram')
            axes[j][0].plot(Lognormal_Pdf_x, Lognormal_Pdf_y1, 'g-', lw=2, label='Rayleigh PDF Fit')
            axes[j][0].set_xlabel('Random Numbers')
            axes[j][0].set_ylabel(f'Probability Density of { string}')
            axes[j][0].set_title(f'{string} Distribution Fit to Input Data')
            axes[j][0].grid()
            axes[j][0].legend()

            axes[j][1].plot(Lognormal_Cdf_data, Lognormal_Cdf_RV_values, marker='*', linestyle='-', color='g', label='Empirical CDF')
            axes[j][1].plot(Lognormal_Cdf_data, Lognormal_Cdf_values, color='r', linestyle='--', label='Rayleigh CDF')
            axes[j][1].set_xlabel('RV data')
            axes[j][1].set_ylabel(f'{string}_CDF')
            axes[j][1].set_title(f'Empirical CDF vs. {string } CDF')
            axes[j][1].grid()
            axes[j][1].legend()

        # Adjust the layout
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=8.0)

        # Show the plot
        plt.show()

        
    elif string == 'Beta':
                    # Create a figure and subplots
        # Create a figure and subplots
        num_cols = len(Experiments_number)
        fig, axes = plt.subplots(num_cols,2, figsize=(12,12))

        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=3, ncols=1)
       

        for j, (n,subfig) in enumerate(zip(Experiments_number,subfigs)):
            
            subfig.subplots_adjust(top=0.85)
            subfig.suptitle(f'For the distribution of { string}, The number of samples is{ n}', fontsize=16, color = "navy", fontweight="bold")


            alpha = 2
            beta = 4
            Beta_Pdf_data, Beta_Pdf_x, Beta_Pdf_y1, Beta_Pdf_y2 = Beta_Pdf_Comparation(n, alpha, beta)
            Beta_Cdf_data, Beta_Cdf_RV_values, Beta_Cdf_x, beta_cdf = Beta_Cdf_Comparation(n, alpha, beta)

            # print(f'---------------------------------------------------------')
            # print(f'For the distribution of { string}, The number of samples is{ n}')


            # Plot data on each subplot
            axes[j][0].hist(Beta_Pdf_data, bins=100, density=True, alpha=0.6, color='b', label='Data Histogram')
            axes[j][0].plot(Beta_Pdf_x, Beta_Pdf_y1, 'g-', lw=2, label='Rayleigh PDF Fit')
            axes[j][0].set_xlabel('Random Numbers')
            # axes[j][0].set_xlim(0, n)

            axes[j][0].set_ylabel(f'Probability Density of { string}')
            axes[j][0].set_title(f'{string} Distribution Fit to Input Data')
            axes[j][0].grid()
            axes[j][0].legend()

            axes[j][1].plot(Beta_Cdf_data, Beta_Cdf_RV_values, marker='*', linestyle='-', color='g', label='Empirical CDF')
            axes[j][1].plot(Beta_Cdf_x, beta_cdf, color='r', linestyle='--', label='Rayleigh CDF')
            axes[j][1].set_xlabel('RV data')
            # axes[j][1].set_xlim(0, n)
            axes[j][1].set_ylabel(f'{string}_CDF')
            axes[j][1].set_title(f'Empirical CDF vs. {string } CDF')
            axes[j][1].grid()
            axes[j][1].legend()

        # Adjust the layout
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=8.0)

        # Show the plot
        plt.show()

        
    elif string == 'Chi-square':
        # Create a figure and subplots
        num_cols = len(Experiments_number)
        fig, axes = plt.subplots(num_cols,2, figsize=(12,12))

        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=3, ncols=1)
       

        for j, (n,subfig) in enumerate(zip(Experiments_number,subfigs)):
            
            subfig.subplots_adjust(top=0.85)
            subfig.suptitle(f'For the distribution of { string}, The number of samples is{ n}', fontsize=16, color = "navy", fontweight="bold")
            k = 9

            chi_square_Pdf_data, chi_square_Pdf_x, chi_square_Pdf_y1, chi_square_Pdf_y2 = chi_square_Pdf_Comparation(k,n)
            chi_square_Cdf_x, chi_square_Cdf_RV_values, chi_square_Cdf = chi_square_Cdf_Comparation(k,n)
            # print(f'---------------------------------------------------------')
            # print(f'For the distribution of { string}, The number of samples is { n}')

            # Plot data on each subplot
            # axes[j].set_title(f'{string} Distribution Fit to Input Data, The number of samples is { n} ')
            axes[j][0].hist(chi_square_Pdf_data, bins=100, density=True, alpha=0.6, color='b', label='Data Histogram')
            axes[j][0].plot(chi_square_Pdf_x, chi_square_Pdf_y1, 'g-', lw=2, label='Rayleigh PDF Fit')
            axes[j][0].set_xlabel('Random Numbers')
            axes[j][0].set_ylabel(f'Probability Density of { string}')
            axes[j][0].set_title(f'{string} Distribution Fit to Input Data')
            axes[j][0].grid()
            axes[j][0].legend()

            axes[j][1].plot(chi_square_Cdf_x, chi_square_Cdf_RV_values, marker='*', linestyle='-', color='g', label='Empirical CDF')
            axes[j][1].plot(chi_square_Cdf_x, chi_square_Cdf, color='r', linestyle='--', label='Rayleigh CDF')
            axes[j][1].set_xlabel('RV data')
            axes[j][1].set_ylabel(f'{string}_CDF')
            axes[j][1].set_title(f'Empirical CDF vs. {string } CDF')
            axes[j][1].grid()
            axes[j][1].legend()

        # Adjust the layout
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=8.0)

        # Show the plot
        plt.show()

        
    elif string == 'Rice':
        # Create a figure and subplots
        num_cols = len(Experiments_number)
        fig, axes = plt.subplots(num_cols,2, figsize=(12,12))

        # create 3x1 subfigs
        subfigs = fig.subfigures(nrows=3, ncols=1)
       

        for j, (n,subfig) in enumerate(zip(Experiments_number,subfigs)):
            
            subfig.subplots_adjust(top=0.85)
            subfig.suptitle(f'For the distribution of { string}, The number of samples is{ n}', fontsize=16, color = "navy", fontweight="bold")
            u = 4
            sigma = 1

            Rice_Pdf_data, Rice_Pdf_x, Rice_Pdf_y2 = Rice_Pdf_Comparation(n, u, sigma)
            Rice_Cdf_data, Rice_Cdf_x, Rice_Cdf_RV_values, Rice_Cdf = Rice_Cdf_Comparation(n, u, sigma)

            # Plot data on each subplot
            axes[j][0].hist(Rice_Pdf_data, bins=100, density=True, alpha=0.6, color='b', label='Data Histogram')
            axes[j][0].plot(Rice_Pdf_x, Rice_Pdf_y2, 'g-', lw=2, label='Rayleigh PDF Fit')
            axes[j][0].set_xlabel('Random Numbers')
            axes[j][0].set_ylabel(f'Probability Density of { string}')
            axes[j][0].set_title(f'{string} Distribution Fit to Input Data')
            axes[j][0].grid()
            axes[j][0].legend()

            axes[j][1].plot(Rice_Cdf_x, Rice_Cdf_RV_values, marker='*', linestyle='-', color='g', label='Empirical CDF')
            axes[j][1].plot(Rice_Cdf_x, Rice_Cdf, color='r', linestyle='--', label='Rayleigh CDF')
            axes[j][1].set_xlabel('RV data')
            axes[j][1].set_ylabel(f'{string}_CDF')
            axes[j][1].set_title(f'Empirical CDF vs. {string } CDF')
            axes[j][1].grid()
            axes[j][1].legend()

        # Adjust the layout
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=8.0)

        # Show the plot
        plt.show()


        # # Save all pictures (plots) as image files
        # for i, ax in enumerate(axes):
        #     ax.figure.savefig(f'plot_{i + 1}.png', dpi=300, bbox_inches='tight')

