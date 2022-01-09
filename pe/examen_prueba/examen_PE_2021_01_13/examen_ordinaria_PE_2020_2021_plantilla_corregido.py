import numpy as np
import matplotlib.pyplot as plt
import stochastic_plots as stoch
import BM_simulators as BM
from scipy.integrate import quad
from scipy import stats
from typing import List, Set, Dict, Tuple, Optional
import seaborn as sns


def simulate_continuous_time_Markov_Chain(
    transition_matrix: np.ndarray,
    lambda_rates: np.ndarray, 
    state_0: int, 
    M: int, 
    t0: float, 
    t1: float,
) -> Tuple[list, list]:

    """ Simulation of a continuous time Markov chain 
    
    Parameters
    ----------
    transition_matrix : 
        Square matrix of transition probabilities between states.
        Rows have to add up to 1.0.
    lambda_rates :
        Rates for each of the states
    state_0 : 
       Initial state encoded as an integer n = 0, 1,... 
    M : 
        Number of trajectories simulated. 
    t0 : 
        Initial time in the simulation.
    t1 : 
        Final time in the simulation.
            
    Returns
    -------

    arrival_times : list
       List of M sublists with the arrival times.
       Each sublist is a the sequence of arrival times in a trajectory
       The first of element of each sublist is t0.
    
    trajectories : list
        List of M sublists.
        Sublist m is trajectory compose of a sequence of states
        of length len(arrival_times[m]).
        All trajectories start from state_0.      

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import examen_ordinaria_PE_2020_2021 as pe
    >>> transition_matrix = [[  0,   1, 0], 
    ...                      [  0,   0, 1],
    ...                      [1/2, 1/2, 0]]
    >>> lambda_rates = [2, 1, 3]
    >>> t0 = 0.0
    >>> t1 = 100.0
    >>> state_0 = 0
    # Simulate and plot a trajectory.
    >>> M = 1 # Number of simulations
    >>> N = 100 # Time steps per simulation
    >>> arrival_times_CTMC, trajectories_CTMC = (
    ...     pe.simulate_continuous_time_Markov_Chain(
    ...     transition_matrix, lambda_rates, 
    ...     state_0, M, t0, t1))
    >>> fig, ax = plt.subplots(1, 1, figsize=(10,5), num=1)
    >>> ax.step(arrival_times_CTMC[0], 
    ...         trajectories_CTMC[0],
    ...         where='post')
    >>> ax.set_ylabel('state')
    >>> ax.set_xlabel('time')
    >>> _ = ax.set_title('Simulation of a continuous-time Markov chain')
    """
    # Initial values
    arrival_times = [[t0] for _ in range(M)]
    trajectories = [[state_0] for _ in range(M)]

    # Computation of each trajectory
    for tr in range(M):
        # Initial values
        t, s = t0, state_0

        # Save current beta value for exponential distribution scale parametrization
        #beta = 1./lambda_rates[s]

        #t = stats.expon.rvs(scale=beta)

        # Simulation with time limit
        while(True):
            # Compute next arrival time
            t += stats.expon.rvs(scale=1./lambda_rates[s])

            if(t >= t1):
                break
            
            # Compute next state and selecting cumsum(P[state]) ≥ random number from uniform [0,1]
            s = np.where(np.cumsum(transition_matrix[s]) >= np.random.uniform(0.,1.))[0][0]

            # Store next state and arrival time
            arrival_times[tr].append(t)
            trajectories[tr].append(s)

    return arrival_times, trajectories

def plot_ctmc_simulation(arrival_times_CTMC, trajectories_CTMC, M):
    """ Plots a CTMC simulation for M trajectories

    Parameters
    -------

    arrival_times : list
       List of M sublists with the arrival times.
       Each sublist is a the sequence of arrival times in a trajectory
       The first of element of each sublist is t0.
    
    trajectories : list
        List of M sublists.
        Sublist m is trajectory compose of a sequence of states
        of length len(arrival_times[m]).
        All trajectories start from state_0.
    
    M : int
        Number of trajectories

    """
    # Subplots parameters
    fig, ax = plt.subplots(M, 1, figsize=(14, 16))
    plt.suptitle(f'Simulation of {M} trajectories for a CTMC', size=18)
    plt.xlabel('t')
    # Select M colors for each trajectory from a determined spectrum
    colors = sns.dark_palette(sns.color_palette("dark:cornflowerblue_r")[0], M, reverse=True)
    # Plot for each trajectory
    for m in range(M):
        ax[m].set_title('M {}'.format(m+1), size=16)
        ax[m].set_ylabel('state')
        ax[m].step(arrival_times_CTMC[m], trajectories_CTMC[m], c=colors[m])
    # Plot padding
    fig.tight_layout(pad=2.)

def price_EU_call(
    S0: float, 
    K: float, 
    r: float, 
    sigma: float, 
    T: float,
) -> float:
    """ Price EU call by numerical quadrature. 
    
    Parameters
    ----------
    S0 : 
        Intial market price of underlying.
    K :
        Strike price of the option.
    r : 
        Risk-free interest rate (anualized).
    sigma : 
        Volatility of the underlyi9ng (anualized).
    T : 
        Lifetime of the optiom (in years).
            
    Returns
    -------
    price : float
        Market price of the option.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import examen_ordinaria_PE_2020_2021 as pe
    >>> S0 = 100.0
    >>> K = 90.0
    >>> r = 0.05
    >>> sigma = 0.3
    >>> T = 2.0
    >>> price_EU_call = pe.price_EU_call(S0, K, r, sigma, T)
    >>> print('Price = {:.4f}'.format(price_EU_call))
    Price = 26.2402
    """

    def integrand(z):
        """ Integrand of a European option. """
        S_T = S0 * np.exp((r - 0.5*sigma**2) * T + sigma * np.sqrt(T) * z)   
        payoff = np.maximum(S_T - K, 0)
        return payoff * stats.norm.pdf(z)

    discount_factor = np.exp(- r * T) 
    R = 10.0
    price_EU_call = discount_factor * quad(integrand, -R, R)[0]
    
    return price_EU_call


def price_EU_call_MC(
    S0: float, 
    K: float, 
    r:float, 
    sigma: float, 
    T: float,
    M: int,
    N: int
) -> Tuple[float, float]:

    """ Price EU call by numerical quadrature. 
    
    Parameters
    ----------
    S0 : 
        Intial market price of underlying.
    K :
        Strike price of the option.
    r : 
        Risk-free interest rate (anualized).
    sigma : 
        Volatility of the underlyi9ng (anualized).
    T : 
        Lifetime of the optiom (in years).
    M :
        Number of simulated trajectories.
    N :        
        Number of timesteps in the simulation.
    Returns
    -------
    price_MC : float
        Monte Carlo estimate of the price of the option
    stdev_MC : float
        Monte Carlo estimate of the standard devuation of price_MC
        
    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import examen_ordinaria_PE_2020_2021 as pe
    >>> S0 = 100.0
    >>> K = 90.0
    >>> r = 0.05
    >>> sigma = 0.3
    >>> T = 2.0
    >>> M = 1000000
    >>> N = 10
    >>> price_EU_call_MC, stdev_EU_call_MC = pe.price_EU_call_MC(S0, K, r, sigma, T, M, N)
    >>> print('Price (MC)= {:.4f} ({:.4f})'.format(price_EU_call_MC, stdev_EU_call_MC))
    """

    return price_MC, stdev_MC
