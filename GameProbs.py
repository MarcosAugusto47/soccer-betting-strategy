import numpy as np
import pandas as pd
from data import load_map

class GameProbs:
    def __init__(self, match_id):
        json_dict = load_map("data\meanSurface.json")
        self.parsed_data = json_dict[match_id]
   
    def build_dataframe(self, nrow=7, ncol=7):
        
        # Create a dataframe with 7 rows and 7 columns filled with np.nan
        df = np.full((nrow, ncol), 0)
        df = pd.DataFrame(df)
        df.columns = [str(number) for number in range(ncol)]
        df.index = [str(number) for number in range(nrow)]

        # Reshape the data_list to match the shape of the DataFrame (7x7)
        reshaped_data = np.transpose(np.array(self.parsed_data).reshape(nrow, ncol))
        
        # Assign the reshaped data to the DataFrame
        df.loc[:, :] = reshaped_data
        self.df = df

        return self.df
