import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import ConstantInputWarning

# Set up matplotlib backend
try:
    import matplotlib
    matplotlib.use('Agg')
except Exception as e:
    print(f"Warning: Visualization setup error: {e}")

# Now import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from standardisation import (
    create_output_directories,
    load_initial_datasets,
    verify_datasets,
    merge_datasets,
    standardize_column_names,
    convert_data_types,
    standardize_numeric_values,
    calculate_correlations,
    save_results,
    create_visualizations,
    get_common_countries,
    filter_and_save_datasets
)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=ConstantInputWarning)

# Create output directories
def create_output_directories():
    """Create necessary output directories"""
    directories = ['processed_data', 'plots', 'plots/advanced']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create directories at startup
create_output_directories()

# Rest of your code follows...

def normalize_column_names(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

def load_initial_datasets():
    try:
        # List of encodings to try
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        datasets = {}
        
        files = {
            'corruption': 'datasets/corruption.csv',
            'democracy': 'datasets/democracy.csv',
            'gdppercapita': 'datasets/gdppercapita.csv',
            'lfpr': 'datasets/lfpr.csv',
            'migration': 'datasets/migration.csv',
            'slavery_2023': 'datasets/slavery_2023.csv',
            'slavery_2018': 'datasets/slavery_2018.csv',
            'slavery_2016': 'datasets/slavery_2016.csv'
        }
        
        for name, file_path in files.items():
            success = False
            for encoding in encodings:
                try:
                    print(f"Trying to load {name} with {encoding} encoding...")
                    df = pd.read_csv(file_path, encoding=encoding)
                    datasets[name] = df
                    print(f"Successfully loaded {name} with {encoding} encoding")
                    success = True
                    break
                except UnicodeDecodeError:
                    continue
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    raise
            
            if not success:
                raise ValueError(f"Failed to load {name} with any encoding")
        
        # Normalize column names for all datasets
        return {name: normalize_column_names(df) for name, df in datasets.items()}
    
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

def verify_datasets(dfs):
    for name, df in dfs.items():
        print(f"\nVerifying dataset: {name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_cols = ['country', 'year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("Null value counts:")
            print(null_counts[null_counts > 0])

def standardize_country_names(df):
    country_mapping = {
        "United States": "USA",
        "United Kingdom": "UK",
        "United States of America": "USA",
        "Great Britain": "UK",
        # Add more mappings as needed
    }
    df['country'] = df['country'].replace(country_mapping)
    return df

def preprocess_lfpr(df):
    try:
        # Ensure required columns exist
        required_cols = ['country', 'year', 'type_lfpr', 'lfpr']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"LFPR dataset missing required columns. Found: {df.columns.tolist()}")
        
        # Pivot LFPR data
        lfpr_pivoted = df.pivot_table(
            index=['country', 'year'],
            columns='type_lfpr',
            values='lfpr'
        ).reset_index()
        
        # Rename columns to lowercase
        lfpr_pivoted.columns = lfpr_pivoted.columns.str.lower()
        
        return lfpr_pivoted
    
    except Exception as e:
        print(f"Error preprocessing LFPR data: {e}")
        raise

def merge_datasets(dfs, target_year):
    try:
        slavery_key = f'slavery_{target_year}'
        if slavery_key not in dfs:
            raise KeyError(f"Missing slavery dataset for year {target_year}")
        
        # Start with standardized slavery dataset
        merged_df = standardize_country_names(dfs[slavery_key].copy())
        
        # Merge with other datasets
        for name, df in dfs.items():
            if name not in ['slavery_2023', 'slavery_2018', 'slavery_2016']:
                print(f"Processing {name} dataset...")
                
                # Standardize country names
                df = standardize_country_names(df)
                
                # Preprocess LFPR if needed
                if name == 'lfpr':
                    df = preprocess_lfpr(df)
                
                # Filter for target year and merge
                year_data = df[df['year'] == target_year]
                if not year_data.empty:
                    merged_df = merged_df.merge(
                        year_data,
                        on='country',
                        how='left',
                        suffixes=('', f'_{name}')
                    )
                    print(f"Merged {name} data: {merged_df.shape}")
                else:
                    print(f"Warning: No data for {name} in year {target_year}")
        
        return merged_df
    
    except Exception as e:
        print(f"Error merging datasets: {e}")
        raise


def main_process():
    try:
        # Create output directory if it doesn't exist
        import os
        os.makedirs('processed_data', exist_ok=True)
        
        # Load and verify datasets
        print("Loading datasets...")
        datasets = load_initial_datasets()
        
        print("\nVerifying datasets...")
        verify_datasets(datasets)
        
        # Process each year
        for year in [2023, 2018, 2016]:
            print(f"\nProcessing year {year}...")
            merged_df = merge_datasets(datasets, year)
            
            # Save merged dataset
            output_path = f'processed_data/merged_{year}.csv'
            merged_df.to_csv(output_path, index=False)
            print(f"Saved merged dataset to {output_path}")
            
            # Print summary statistics
            print(f"Final shape: {merged_df.shape}")
            print("Columns:", merged_df.columns.tolist())
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

def load_merged_datasets():
    """Load all merged datasets and return basic information about them"""
    years = [2016, 2018, 2023]
    datasets = {}
    
    for year in years:
        try:
            df = pd.read_csv(f'processed_data/merged_{year}.csv')
            datasets[year] = df
            
            print(f"\nMerged Dataset {year}:")
            print("-" * 50)
            print(f"Shape: {df.shape}")
            print("\nFirst few rows:")
            print(df.head())
            print("\nColumns:")
            print(df.columns.tolist())
            print("\nMissing values:")
            print(df.isnull().sum()[df.isnull().sum() > 0])
            print("\nSummary statistics:")
            print(df.describe())
            
            # Check which countries are included
            print(f"\nNumber of countries: {df['country'].nunique()}")
            print("\nSample of countries:")
            print(df['country'].sample(min(5, len(df['country']))).tolist())
            
        except FileNotFoundError:
            print(f"Warning: Could not find merged_{year}.csv in processed_data directory")
        except Exception as e:
            print(f"Error processing {year} dataset: {e}")
    
    return datasets

def compare_datasets(datasets):
    """Compare the different year datasets"""
    try:
        if not datasets:
            print("No datasets available for comparison")
            return
            
        years = list(datasets.keys())
        
        print("\nDataset Comparison:")
        print("-" * 50)
        
        # Compare number of countries
        print("\nNumber of countries in each dataset:")
        for year in years:
            print(f"{year}: {datasets[year]['country'].nunique()} countries")
        
        # Find common countries
        common_countries = set(datasets[years[0]]['country'])
        for year in years[1:]:
            common_countries = common_countries.intersection(set(datasets[year]['country']))
        
        print(f"\nNumber of common countries across all years: {len(common_countries)}")
        print("\nSample of common countries:")
        print(list(common_countries)[:5])
        
    except Exception as e:
        print(f"Error comparing datasets: {e}")

def load_merged_datasets():
    """Load all merged datasets"""
    try:
        datasets = {}
        for year in [2016, 2018, 2023]:
            df = pd.read_csv(f'processed_data/merged_{year}.csv')
            datasets[year] = df
        return datasets
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None

def get_common_countries(datasets):
    """Get list of countries common to all datasets"""
    try:
        # Get sets of countries from each dataset
        country_sets = [set(df['country'].unique()) for df in datasets.values()]
        
        # Get intersection of all sets
        common_countries = set.intersection(*country_sets)
        
        print(f"Number of common countries: {len(common_countries)}")
        print("\nSample of common countries:")
        print(sorted(list(common_countries))[:10])
        
        return sorted(list(common_countries))
    except Exception as e:
        print(f"Error getting common countries: {e}")
        return None

def filter_and_save_datasets(datasets, common_countries):
    """Filter datasets to include only common countries and save"""
    try:
        filtered_datasets = {}
        
        for year, df in datasets.items():
            # Filter for common countries
            filtered_df = df[df['country'].isin(common_countries)].copy()
            
            # Sort by country name for consistency
            filtered_df = filtered_df.sort_values('country').reset_index(drop=True)
            
            # Save filtered dataset
            output_path = f'processed_data/filtered_merged_{year}.csv'
            filtered_df.to_csv(output_path, index=False)
            
            filtered_datasets[year] = filtered_df
            
            print(f"\nDataset {year}:")
            print(f"Original shape: {df.shape}")
            print(f"Filtered shape: {filtered_df.shape}")
            
            # Verify all countries are present
            missing_countries = set(common_countries) - set(filtered_df['country'])
            if missing_countries:
                print(f"Warning: Missing countries in {year}: {missing_countries}")
    
        return filtered_datasets
    except Exception as e:
        print(f"Error filtering datasets: {e}")
        return None

def verify_filtered_datasets(filtered_datasets):
    """Verify that filtered datasets contain the same countries"""
    try:
        print("\nVerifying filtered datasets:")
        
        # Check country counts
        for year, df in filtered_datasets.items():
            print(f"\n{year} dataset:")
            print(f"Number of countries: {df['country'].nunique()}")
            print("First 5 countries:")
            print(df['country'].head().tolist())
            
        # Verify columns
        print("\nColumns in each dataset:")
        for year, df in filtered_datasets.items():
            print(f"\n{year} columns:")
            print(df.columns.tolist())
            
    except Exception as e:
        print(f"Error verifying datasets: {e}")

def main_analyze():
    try:
        # Load original merged datasets
        print("Loading datasets...")
        datasets = load_merged_datasets()
        
        if datasets:
            # Get common countries
            print("\nFinding common countries...")
            common_countries = get_common_countries(datasets)
            
            if common_countries:
                # Filter and save datasets
                print("\nFiltering datasets...")
                filtered_datasets = filter_and_save_datasets(datasets, common_countries)
                
                if filtered_datasets:
                    # Verify filtered datasets
                    verify_filtered_datasets(filtered_datasets)
                    
                    print("\nSuccess! Filtered datasets have been saved to processed_data/")
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        
def process_initial_data():
    """Initial data processing function - runs first"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs('processed_data', exist_ok=True)
        
        # Load and verify datasets
        print("Loading datasets...")
        datasets = load_initial_datasets()
        
        print("\nVerifying datasets...")
        verify_datasets(datasets)
        
        # Process each year
        for year in [2023, 2018, 2016]:
            print(f"\nProcessing year {year}...")
            merged_df = merge_datasets(datasets, year)
            
            # Save merged dataset
            output_path = f'processed_data/merged_{year}.csv'
            merged_df.to_csv(output_path, index=False)
            print(f"Saved merged dataset to {output_path}")
            
            # Print summary statistics
            print(f"Final shape: {merged_df.shape}")
            print("Columns:", merged_df.columns.tolist())
        
        return True
    
    except Exception as e:
        print(f"Error in initial data processing: {e}")
        return False

def analyze_processed_data():
    """Analysis function - runs after initial processing"""
    try:
        print("\nLoading and analyzing merged datasets...")
        datasets = load_merged_datasets()
        if datasets:
            compare_datasets(datasets)
            
            # Get common countries and filter datasets
            common_countries = get_common_countries(datasets)
            if common_countries:
                filtered_datasets = filter_and_save_datasets(datasets, common_countries)
                if filtered_datasets:
                    verify_filtered_datasets(filtered_datasets)
    
    except Exception as e:
        print(f"Error in data analysis: {e}")

def analyze_lfpr_correlations(datasets):
    """
    Enhanced LFPR correlation analysis with multiple testing correction
    """
    for year, df in datasets.items():
        print(f"\nYear {year}:")
        
        # Calculate correlations
        correlations = calculate_correlations(df)
        
        # Apply multiple testing corrections
        corrected_results = apply_multiple_testing_corrections(correlations)
        
        # Create visualizations
        create_correlation_plots({year: df})
        
        # Perform stratified analysis
        stratified_results = perform_stratified_analysis({year: df})
        
        # Print results
        print("\nCorrelation Results with Multiple Testing Corrections:")
        for var, corr in corrected_results['correlations'].items():
            print(f"\n{var}:")
            print(f"  r = {corr:.3f}")
            print(f"  Original p-value: {corrected_results['p_values'][var]:.3e}")
            print(f"  Bonferroni p-value: {corrected_results['bonferroni_corrected_p'][var]:.3e}")
            print(f"  Holm p-value: {corrected_results['holm_corrected_p'][var]:.3e}")
            print(f"  FDR p-value: {corrected_results['fdr_bh_corrected_p'][var]:.3e}")

if __name__ == "__main__":
    try:
        # Previous code remains the same until processed_datasets is created
        
        if processed_datasets:
            print("\nStep 6: Creating visualizations and performing analyses...")
            
            # Create basic visualizations
            create_visualizations(processed_datasets)
            
            # Create correlation plots
            create_correlation_plots(processed_datasets)
            
            # Perform stratified analysis
            stratified_results = perform_stratified_analysis(processed_datasets)
            
            # Apply multiple testing corrections to correlation results
            for year, df in processed_datasets.items():
                correlations, p_values = calculate_correlations(df)
                corrected_results = apply_multiple_testing_corrections({
                    'correlations': correlations,
                    'p_values': p_values
                })
                
                # Print results
                print(f"\nResults for {year} with multiple testing corrections:")
                for var in correlations.keys():
                    print(f"\n{var}:")
                    print(f"Correlation: {correlations[var]:.3f}")
                    print(f"Original p-value: {p_values[var]:.3e}")
                    for method in ['bonferroni', 'holm', 'fdr_bh']:
                        print(f"{method} corrected p-value: {corrected_results[f'{method}_corrected_p'][var]:.3e}")
            
            print("\nAnalysis complete! Check the 'processed_data' and 'plots' directories for results.")
        else:
            print("No datasets were successfully processed")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        print(traceback.format_exc())