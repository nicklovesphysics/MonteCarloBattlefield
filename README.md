# MonteCarloBattlefield
An optimizer to refine my strategy for the Maroon Capital 2026-27 Challenge problem.

score_calc computes individual scores based on the problem statement.

compute_gradient computes gradient for gradient descent.

save_p2_players saves an array of players for player 2. the weights are controlled in the 'p = ' argument for random.choice().

The rest is just win/loss calculation, optimization, etc. 


I've found my most successful solution has been trained with uniformly distributed data. the only model it's lost to has been one with a normal distribution of soldiers around 5. This model loses to every other model type, though. so, i'm further training the model just a bit to attempt to beat this normal distribution around 5-6 a tad bit more. 