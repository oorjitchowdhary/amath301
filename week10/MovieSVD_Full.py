import numpy as np
import pandas as pd
import os

##### Complete Movie Recommendation algorithm

# First, load in the data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'cleaned_data.csv'))

# Save the original data. 
data_py = np.copy(data.to_numpy()) # Save as a numpy array because we will
                                    # use that later.
data_py = np.array(data_py[:, 1:], dtype=float) # remove student numbers

## Shift the data so each row has zero mean and fill in missing with zeros
avg_user_ratings = np.nanmean(data_py, axis=1)
data_py -= avg_user_ratings.reshape(-1, 1)

# Now we will run the algorithm 
r = 1 # Set the rank to 1, we will do rank-1 updates
tol = 1e-8 # We will stop when our successive approximations are this close

Ak = np.nan_to_num(data_py) # Initial guess
for k in range(100000):
    U, S, Vt = np.linalg.svd(Ak, full_matrices = False)
    S = np.diag(S)
    Akplus1 = (U[:, :r]@S[:r, :r])@Vt[:r, :] # Redefine Akplus1
    # Replace the values in the low-rank approximation with the values we know
    Akplus1[~np.isnan(data_py)] = data_py[~np.isnan(data_py)]
    if np.linalg.norm(Ak - Akplus1)<tol:
         break
    Ak = Akplus1


A_final = Akplus1 + avg_user_ratings.reshape(-1,1) # Add back in the row averages

# Set index to student numbers
data = data.set_index('StudentNo') # Index by student number

final_data = pd.DataFrame(data=A_final, columns=data.columns, index=data.index)
final_data.to_csv('movie_recommendations.csv', index=True)
