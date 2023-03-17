import numpy as np
import pandas as pd
import os

# First, load in the complete data
data_complete = pd.read_csv(os.path.join(os.path.dirname(__file__), 'movie_recommendations.csv'))
# Index by student number
data_complete = data_complete.set_index('StudentNo') 

# Then, load in the original data
data_incomplete = pd.read_csv(os.path.join(os.path.dirname(__file__), 'cleaned_data.csv'))
# Index by student number
data_incomplete = data_incomplete.set_index('StudentNo') 

# Enter your student ID as a string
# Enter your student ID as a string
student_ID = '2235352'

# Find your ratings
raw_data = data_incomplete.loc[[student_ID]]
ratings_data = data_complete.loc[[student_ID]]

for (columnName, columnData) in raw_data.iteritems():
    if np.isnan(columnData[0]): # If your rating was NaN
        print('%.2f'%ratings_data[columnName][0] + ' is your predicted rating for '+str(columnName))
