import numpy as np
import pandas as pd
from itertools import chain
from data import load_map


class PreProcess:
    def __init__(self, match_id):
        self.match_id = match_id
        self.parsed_data = self.preprocess()

    def preprocess(self):
        json_dict = load_map("data/meanSurface-new.json")
        self.parsed_data = json_dict[self.match_id]
        return self.parsed_data

class PreprocessKreiner:
    def __init__(self, match_id):
        self.match_id = match_id
        self.parsed_data = self.preprocess()

    @staticmethod
    def create_custom_matrix(size, below_diag, diag, above_diag):
        matrix = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                if i > j:
                    matrix[i, j] = above_diag / 21
                elif i == j:
                    matrix[i, j] = diag / 7
                else:
                    matrix[i, j] = below_diag / 21
        
        return matrix

    def preprocess(self):
        json_dict = load_map("data/baseline_kreiner.json")
        parsed_data = json_dict[self.match_id]
        
        size = 7
        below_diag = parsed_data[0]
        diag = parsed_data[1]
        above_diag = parsed_data[2]

        matrix = self.create_custom_matrix(size, below_diag, diag, above_diag)

        self.parsed_data = list(chain(*matrix))

        return self.parsed_data

class GameProbs:
    def __init__(self, preprocess_method):
        self.preprocess_method = preprocess_method

    def build_dataframe(self, nrow=7, ncol=7):
        
        # Create a dataframe with 7 rows and 7 columns filled with np.nan
        df = np.full((nrow, ncol), 0)
        df = pd.DataFrame(df)
        df.columns = [str(number) for number in range(ncol)]
        df.index = [str(number) for number in range(nrow)]

        # Reshape the data_list to match the shape of the DataFrame (7x7)
        reshaped_data = np.transpose(np.array(self.preprocess_method.parsed_data).reshape(nrow, ncol))
        # Assign the reshaped data to the DataFrame
        df.loc[:, :] = reshaped_data
        self.df = df

        return self.df
