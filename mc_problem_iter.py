import mcmc as mc
import numpy as np
from tqdm import tqdm

#optimal_p1 = mc.optimize_strategy()
#print("Optimal P1 Soldier Allocation for expected probability:", optimal_p1)
#print("Total Soldiers Allocated:", np.sum(optimal_p1))

#50ke
p1_guess = [4,6,8,11,14,0,18,21,0,18]       #going with this for now. Training into p1_final, which is going to be trained with normal data (centered around 5)
p1_uniform = [0,1,2,11,12,16,14,15,15,14]
optimal_p1 = [ 4,  6,  8, 11, 14,  0, 18, 20,  2, 17]

#1000e

normal = [ 9, 10,  11, 12, 12, 15, 23,  2,  2,  4]
guess = [4,6,8,11,15,0,18,20,0,18]
uni = [0,1, 3,11,12,14,14,14,15,16]




p2 = mc.save_p2_players(10000)

wins = []

#for i in range(len(p2[0])):

    #wins.append(mc.win_score_calc(optimal_p1, p2[i]))

wins.append(mc.win_score_calc(np.array(optimal_p1), np.array(normal)))
print(optimal_p1)

print(np.mean(wins))
