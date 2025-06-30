import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import random

"""
CONTINUOUS-TIME MARKOV CHAIN FOR SIR MODEL
Mathematical Foundations and Implementation
"""

# =============================================================================
# PART 1: INDIVIDUAL-LEVEL CTMC (Single Person)
# =============================================================================

def individual_transition_matrix(beta_effective, gamma):
    """
    For a single individual, the infinitesimal generator matrix Q
    where Q[i,j] is the rate of transition from state i to state j
    
    States: 0=S, 1=I, 2=R
    
    Q = [[-beta_eff,  beta_eff,    0    ]
         [    0,        -gamma,   gamma ]
         [    0,          0,        0   ]]
    
    beta_effective = beta * I(t) / N (depends on current epidemic state)
    """
    Q = np.array([
        [-beta_effective, beta_effective, 0],
        [0, -gamma, gamma],
        [0, 0, 0]
    ])
    return Q

def demonstrate_individual_ctmc():
    """Demonstrate how an individual transitions through S->I->R"""
    
    # Parameters
    gamma = 1/5  # Recovery rate (5 day infectious period)
    I_current = 100  # Current number of infected
    N = 10000    # Total population
    beta = 0.3   # Base transmission rate
    
    beta_eff = beta * I_current / N
    
    Q = individual_transition_matrix(beta_eff, gamma)
    
    print("Individual Transition Rate Matrix Q:")
    print("States: 0=S, 1=I, 2=R")
    print(Q)
    print()
    
    # The diagonal elements are negative (exit rates from each state)
    print("Exit rates from each state:")
    print(f"From S: {-Q[0,0]:.4f} per day")
    print(f"From I: {-Q[1,1]:.4f} per day") 
    print(f"From R: {-Q[2,2]:.4f} per day")
    print()
    
    # Expected time in each state (before transition)
    print("Expected time in each state:")
    print(f"In S (before infection): {1/beta_eff:.2f} days") 
    print(f"In I (infectious period): {1/gamma:.2f} days")
    print("In R: Forever (absorbing state)")

# =============================================================================
# PART 2: POPULATION-LEVEL STATE SPACE
# =============================================================================

def population_state_transitions(S, I, R, beta, gamma, N):
    """
    At population level, state is (S, I, R) where S + I + R = N
    
    From state (S, I, R), two things can happen:
    1. Infection event: (S, I, R) -> (S-1, I+1, R) at rate beta*S*I/N
    2. Recovery event: (S, I, R) -> (S, I-1, R+1) at rate gamma*I
    
    Returns: (rates, new_states)
    """
    
    # Possible transitions and their rates
    transitions = []
    
    if S > 0 and I > 0:  # Infection can occur
        infection_rate = beta * S * I / N
        new_state_infection = (S-1, I+1, R)
        transitions.append((infection_rate, new_state_infection, "infection"))
    
    if I > 0:  # Recovery can occur
        recovery_rate = gamma * I
        new_state_recovery = (S, I-1, R+1)
        transitions.append((recovery_rate, new_state_recovery, "recovery"))
    
    return transitions

def demonstrate_population_transitions():
    """Show how population state transitions work"""
    
    # Example state
    S, I, R = 990, 10, 0
    N = S + I + R
    beta, gamma = 0.3, 0.2
    
    transitions = population_state_transitions(S, I, R, beta, gamma, N)
    
    print(f"Current state: S={S}, I={I}, R={R}")
    print("Possible transitions:")
    
    total_rate = 0
    for rate, new_state, event_type in transitions:
        total_rate += rate
        print(f"  {event_type}: rate={rate:.4f} -> {new_state}")
    
    print(f"Total transition rate: {total_rate:.4f}")
    print(f"Expected time to next event: {1/total_rate:.4f} days")

# =============================================================================
# PART 3: GILLESPIE ALGORITHM FOUNDATION
# =============================================================================

def next_reaction_time(total_rate):
    """
    Time to next reaction is exponentially distributed
    τ ~ Exponential(λ) where λ is total rate
    """
    if total_rate == 0:
        return float('inf')
    return -np.log(random.random()) / total_rate

def choose_reaction(rates):
    """
    Choose which reaction occurs based on rates
    Uses the "direct method" - probability proportional to rate
    """
    total_rate = sum(rates)
    if total_rate == 0:
        return None
    
    # Generate random number between 0 and total_rate
    r = random.random() * total_rate
    
    # Find which reaction this corresponds to
    cumulative = 0
    for i, rate in enumerate(rates):
        cumulative += rate
        if r <= cumulative:
            return i
    
    return len(rates) - 1  # Should never reach here

def gillespie_step(S, I, R, beta, gamma, N, current_time):
    """
    Perform one step of the Gillespie algorithm
    Returns: (new_S, new_I, new_R, new_time, event_type)
    """
    
    # Get all possible transitions
    transitions = population_state_transitions(S, I, R, beta, gamma, N)
    
    if not transitions:  # No more transitions possible
        return S, I, R, float('inf'), "end"
    
    # Extract rates and new states
    rates = [t[0] for t in transitions]
    new_states = [t[1] for t in transitions]
    event_types = [t[2] for t in transitions]
    
    # Calculate time to next event
    total_rate = sum(rates)
    tau = next_reaction_time(total_rate)
    new_time = current_time + tau
    
    # Choose which event occurs
    chosen_reaction = choose_reaction(rates)
    new_S, new_I, new_R = new_states[chosen_reaction]
    event_type = event_types[chosen_reaction]
    
    return new_S, new_I, new_R, new_time, event_type

def demonstrate_gillespie_steps():
    """Show a few steps of the Gillespie algorithm"""
    
    # Initial conditions
    S, I, R = 995, 5, 0
    N = S + I + R
    beta, gamma = 0.3, 0.2
    t = 0.0
    
    print("Gillespie Algorithm Steps:")
    print(f"Initial: t={t:.3f}, S={S}, I={I}, R={R}")
    
    for step in range(5):
        S, I, R, t, event = gillespie_step(S, I, R, beta, gamma, N, t)
        if t == float('inf'):
            print("Epidemic ended (no more infected individuals)")
            break
        print(f"Step {step+1}: t={t:.3f}, S={S}, I={I}, R={R}, event={event}")

# =============================================================================
# PART 4: THEORETICAL CONNECTIONS
# =============================================================================

def theoretical_insights():
    """
    Connect CTMC formulation to differential equations
    """
    print("THEORETICAL CONNECTIONS:")
    print("=" * 50)
    
    print("1. MEAN-FIELD APPROXIMATION:")
    print("   As N -> ∞, the stochastic process converges to ODEs:")
    print("   dS/dt = -βSI/N")
    print("   dI/dt = βSI/N - γI") 
    print("   dR/dt = γI")
    print()
    
    print("2. VARIANCE AROUND DETERMINISTIC SOLUTION:")
    print("   For finite N, there's randomness around the ODE solution")
    print("   Variance ∝ 1/N (law of large numbers)")
    print()
    
    print("3. EXTINCTION PROBABILITY:")
    print("   Small outbreaks can die out randomly even if R₀ > 1")
    print("   This is captured by CTMC but not by ODEs")
    print()
    
    print("4. FINAL SIZE DISTRIBUTION:")
    print("   CTMC gives full distribution of final epidemic size")
    print("   ODEs only give expected final size")

# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================

if __name__ == "__main__":
    print("CONTINUOUS-TIME MARKOV CHAIN SIR MODEL")
    print("=" * 50)
    print()
    
    print("PART 1: Individual Level Transitions")
    print("-" * 40)
    demonstrate_individual_ctmc()
    print()
    
    print("PART 2: Population Level Transitions") 
    print("-" * 40)
    demonstrate_population_transitions()
    print()
    
    print("PART 3: Gillespie Algorithm Steps")
    print("-" * 40)
    random.seed(42)  # For reproducible results
    demonstrate_gillespie_steps()
    print()
    
    print("PART 4: Theoretical Insights")
    print("-" * 40)
    theoretical_insights()