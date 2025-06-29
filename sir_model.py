import numpy as np

def sir_model(S, I, R, beta, gamma):
    """
    Simulate one step of the SIR model.
    
    Parameters:
    S (int): Number of susceptible individuals
    I (int): Number of infected individuals
    R (int): Number of recovered individuals
    beta (float): Infection rate
    gamma (float): Recovery rate
    
    Returns:
    tuple: Updated counts of S, I, and R
    """
    new_infections = beta * S * I / (S + I + R)
    new_recoveries = gamma * I
    
    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    
    return S_new, I_new, R_new

def simulate_sir(S0, I0, R0, beta, gamma, days):
    """
    Simulate the SIR model over a number of days.
    
    Parameters:
    S0 (int): Initial number of susceptible individuals
    I0 (int): Initial number of infected individuals
    R0 (int): Initial number of recovered individuals
    beta (float): Infection rate
    gamma (float): Recovery rate
    days (int): Number of days to simulate
    
    Returns:
    list: List of tuples containing (S, I, R) for each day
    """
    results = []
    S, I, R = S0, I0, R0
    
    for day in range(days):
        results.append((S, I, R))
        S, I, R = sir_model(S, I, R, beta, gamma)
    
    return results

def main():
    # Initial conditions
    S0 = 0.99  # Susceptible individuals as proportion of the population
    I0 = 0.01   # Infected individuals as proportion of the population
    R0 = 0    # Recovered individuals as proportion of the population
    beta = 0.3  # Infection rate
    gamma = 0.1  # Recovery rate
    days = 30   # Number of days to simulate
    
    results = simulate_sir(S0, I0, R0, beta, gamma, days)
    
    for day, (S, I, R) in enumerate(results):
        print(f"Day {day}: S={S}, I={I}, R={R}")

if __name__ == "__main__":
    main()