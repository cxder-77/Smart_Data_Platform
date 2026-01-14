import pandas as pd
import io

class DataValidator:
    @staticmethod
    def validate(file):
        try:
            # Try reading a small portion
            pd.read_csv(io.BytesIO(file.getvalue()), nrows=5)
            return True, "File is valid CSV."
        except Exception:
            return False, "Invalid or corrupted CSV file."
