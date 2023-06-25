import pandas as pd

class ReadData:
    def __init__(self) -> None:
        self.data_path = None
    
    def get_data(self, data_path: str) -> pd.DataFrame:
        
        """
        > Creates pandas data frame from path to csv file

        Inputs:
            - data_path : A string path to the csv file
        Returns:
            - df: A pandas dataframe
        """
    
        self.data_path = data_path
        df = pd.read_csv(self.data_path)

        return df