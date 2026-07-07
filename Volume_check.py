import pandas as pd
import glob
import os

INPUT_FOLDER = "../stock_data"
REPORT_NAME = "data_audit_report.csv"

def main():
    # Find all CSV files in the target directory
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

    if not csv_files:
        print(f"\nCRITICAL ERROR: No CSV files found in the '{INPUT_FOLDER}' directory.")
        return

    audit_data = []

    print(f"\nAuditing {len(csv_files)} files in the '{INPUT_FOLDER}' folder...")

    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract the stock ticker name from the filename (e.g., "AAPL.csv" -> "AAPL")
            ticker = os.path.basename(file_path).replace('.csv', '')

            # Count missing values (NaNs) for every single column
            nan_counts = df.isnull().sum().to_dict()

            # Create a dictionary representing this specific stock's audit row
            row = {
                'Ticker': ticker,
                'Total_Rows': len(df)
            }
            # Append the NaN counts to our dictionary
            row.update(nan_counts)

            audit_data.append(row)

        except Exception as e:
            print(f"  [!] Could not audit {file_path} due to error: {e}")

    # Safety check
    if not audit_data:
        print("\nERROR: Could not process any files. Audit failed.")
        return

    # Convert the list of dictionaries into a Pandas DataFrame
    audit_df = pd.DataFrame(audit_data)

    # Reorder columns
    cols = ['Ticker', 'Total_Rows'] + [c for c in audit_df.columns if c not in ['Ticker', 'Total_Rows']]
    audit_df = audit_df[cols]

    audit_df.to_csv(REPORT_NAME, index=False)

    print(f"\nAudit Complete! Checked {len(audit_df)} stocks.")
    print(f"\nThe report was successfully saved as: {REPORT_NAME}")


if __name__ == "__main__":
    main()