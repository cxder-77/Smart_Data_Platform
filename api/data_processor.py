import pandas as pd

class DataProcessor:
    def __init__(self, file):
        self.df = pd.read_csv(file)

    def get_summary(self):
        summary = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "missing_values": int(self.df.isnull().sum().sum()),
            "data_types": self.df.dtypes.astype(str).to_dict()
        }
        return summary

    def get_dataframe(self):
        return self.df

    def sort_dataframe(self, column, ascending=True):
        if column in self.df.columns:
            return self.df.sort_values(by=column, ascending=ascending)
        return self.df

    def get_numeric_summary(self):
        return self.df.describe()
