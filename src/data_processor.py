import wbgapi as wb
import pandas as pd
import os

class DataProcessor:
    def __init__(self, country_code='NGA'):
        self.country_code = country_code
        self.indicators = {
            'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
            'FP.CPI.TOTL.ZG': 'inflation_rate'
        }

    def fetch_world_bank_data(self, year_range=range(2020, 2026)):
        """Fetches macro-economic indicators."""
        print(f"Fetching data for {self.country_code}...")
        try:
            df = wb.data.DataFrame(self.indicators.keys(), self.country_code, time=year_range, labels=True)
            # Cleanup: rename columns and reset index
            df = df.reset_index()
            # Basic transformation to get a clean Year/Indicator mapping
            return df
        except Exception as e:
            print(f"Error fetching from World Bank: {e}")
            return None

    def prepare_lending_data(self, raw_path):
        """
        Placeholder for Home Credit data processing.
        This will later include the complex joins and aggregations.
        """
        if not os.path.exists(raw_path):
            return "Raw data file not found. Please download Home Credit data to data/raw/"
        
        # We will add LightGBM-specific aggregations here next
        pass

if __name__ == "__main__":
    processor = DataProcessor()
    macro_data = processor.fetch_world_bank_data()
    if macro_data is not None:
        print("Macro Data Sample:")
        print(macro_data.head())