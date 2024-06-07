
import pandas as pd

# Specify the path to your Excel file
excel_file = "C:/Users/hp/Desktop/pp/Book1.xlsx"

# Read the Excel file into a DataFrame

df = pd.read_excel(excel_file, sheet_name="Sheet1")

# Save the DataFrame to a CSV file
csv_file = "output_file.csv"
df.to_csv(csv_file, index=False)

print(f"Data successfully written to {csv_file}")
