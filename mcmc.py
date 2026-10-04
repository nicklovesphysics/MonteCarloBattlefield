import numpy as np
from tqdm import tqdm
import scipy.stats as stats

battlefields = np.arange(1,11)

p2_prob_density = np.array([
    0.02,  
    0.03, 
    0.04, 
    0.07, 
    0.10,  
    0.16,  
    0.13,  
    0.15,  
    0.18,  
    0.12   
])

bfsix = 5       #index of bf6

# most efficient way I know to initialize random arrays where normal distributions around my "thought - of best solution" are comsidered:

rng = np.random.default_rng() #put in random arrays and test and find strategy that works.

n_samples = 1000

def score_calc(p1, p2, k=0.5):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    # Probabilities of winning each battlefield
    w1 = 1.0 / (1.0 + np.exp(-k * (p1 - p2)))
    w2 = 1.0 - w1

    p1_mult = np.ones(10, dtype=float)
    p2_mult = np.ones(10, dtype=float)

    # adjacent bonus
    for i in range(len(battlefields) - 1):
        adj_p1 = w1[i] * w1[i+1]
        adj_p2 = w2[i] * w2[i+1]
        
        p1_mult[i] = max(p1_mult[i], 1.0 + 0.5 * adj_p1)
        p1_mult[i+1] = max(p1_mult[i+1], 1.0 + 0.5 * adj_p1)

        p2_mult[i] = max(p2_mult[i], 1.0 + 0.5 * adj_p2)
        p2_mult[i+1] = max(p2_mult[i+1], 1.0 + 0.5 * adj_p2)

    p1_raw = np.sum(battlefields * w1)
    p2_raw = np.sum(battlefields * w2)

    # multiple of 5 smooth approximation
    p1_mult *= (1.0 + 0.4 * 0.5 * (1.0 + np.cos(2 * np.pi * p1_raw / 5.0)))
    p2_mult *= (1.0 + 0.4 * 0.5 * (1.0 + np.cos(2 * np.pi * p2_raw / 5.0)))

    #BF6 multiplier
    p1_mult *= (1.0 + 0.1 * w1[bfsix])
    p2_mult *= (1.0 + 0.1 * w2[bfsix])

    p1_final = np.sum(battlefields * w1 * p1_mult)
    p2_final = np.sum(battlefields * w2 * p2_mult)

    return p1_final - p2_final

def compute_gradient(p1, p2_batch, h=1e-4):
    
    grad = np.zeros_like(p1, dtype=float)
    
    for i in range(len(p1)):
        p1_plus = p1.copy()
        p1_plus[i] += h
        
        p1_minus = p1.copy()
        p1_minus[i] -= h
        
        
        margins_plus = np.mean([score_calc(p1_plus, p2) for p2 in p2_batch])
        margins_minus = np.mean([score_calc(p1_minus, p2) for p2 in p2_batch])
        
        # Central difference formula
        grad[i] = (margins_plus - margins_minus) / (2 * h)
        
    return grad

#saving an array of training data in folder
def save_p2_players(n_samples, battle_fields = battlefields):
    probabilities = stats.norm.pdf(battle_fields, loc=9, scale=1.25)
    probabilities /= np.sum(probabilities)

    save_array = np.zeros((n_samples, len(battle_fields)))

    for i in tqdm(range(n_samples)):
        p2_choices = rng.choice(battlefields, size = 100, replace= True)#, p = p2_prob_density)#, p = p2_prob_density)       #easily add in probability distribution for guessing strategies with p argument.
        p2_soldiers = np.bincount(p2_choices, minlength=11)[1:]

        save_array[i] = p2_soldiers
    return save_array

def project_onto_budget(p1, total_budget=100.0):
    #just scales sum to 100 players. 
    p1 = np.maximum(p1, 0.0)
    current_sum = np.sum(p1)
    if current_sum == 0:
        return np.ones_like(p1) * (total_budget / len(p1)) 
    return p1 * (total_budget / current_sum)


def error_gradient(p1_score, p2_score):
    target = p2_score + 1

    mse = np.mean((target - p1_score))
    return -2*mse

def optimize_strategy(epochs=1000, batch_size=50, lr=0.5):
    # Initialize P1 with an even distribution of 10 soldiers per battlefield
    p1 = np.ones(10, dtype=float) * 10.0
    
    # Generate training dataset of opponents
    p2_data = save_p2_players(1000)
    
    print("Training...")
    for epoch in tqdm(range(epochs)):
        # Sample a random batch of opponents
        batch_indices = np.random.choice(len(p2_data), size=batch_size, replace=False)
        p2_batch = p2_data[batch_indices]
        
        # Compute gradient wrt score advantage
        grad = compute_gradient(p1, p2_batch)
        
        # Gradient Ascent step
        p1 += lr * grad
        
        # Enforce non-negativity and 100 soldier sum
        p1 = project_onto_budget(p1, total_budget=100.0)

    p1_int = np.round(p1).astype(int)
    diff = 100 - np.sum(p1_int)
    p1_int[np.argmax(p1)] += diff
    
    return p1_int

    

        
def win_score_calc(p1,p2, bf_six = bfsix):

    difference = p1-p2
    outcome = np.sign(difference) #-1 means player 2 won, 1 means player 1 won, 0 means tie.

    p1_raw = np.sum(p1)
    p2_raw = np.sum(p2)
    p1_multipliers = np.ones_like(p1, dtype = float)
    p2_multipliers = np.ones_like(p2, dtype = float)


    for i in range(len(outcome)-1):


        if outcome[i] == 1 and outcome[i+1] ==1:
            p1_multipliers[i] = 1.5
            p1_multipliers[i+1] = 1.5

        if outcome[i] == -1 and outcome[i+1] == -1:
            p2_multipliers[i] = 1.5
            p2_multipliers[i+1] = 1.5

    p1_raw = np.sum(battlefields[outcome == 1])
    p2_raw = np.sum(battlefields[outcome == -1])



    #p1 multiple of 5 multiplier (1.4)

    if p1_raw%5 == 0:
        p1_multipliers *= 1.4

    #p2 multiple of 5 multiplier (1.4)

    if p2_raw%5 == 0:
        p2_multipliers *= 1.4

    #whoever won 'field 6' gets 1.1x multiplier (if a win happened):

    if outcome[bf_six] < 0:
        p2_multipliers*=1.1

    elif outcome[bf_six] > 0:
        p1_multipliers*=1.1


    p1_final = np.sum((p1_multipliers[outcome == 1]*battlefields[outcome == 1]))
    p2_final = np.sum((p2_multipliers[outcome == -1]*battlefields[outcome == -1]))

    #print(f'sums of points = ',p1_final, p2_final)

    if p1_final < p2_final:
        return -1

    elif p1_final > p2_final:
        return 1

    else:
        return 0


        
        

