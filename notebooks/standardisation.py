import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import ConstantInputWarning

# Suppress specific warnings
warnings.filterwarnings('ignore', category=ConstantInputWarning)

def standardize_column_names(df, year):
    """
    Standardize column names based on the year of the dataset
    """
    # Print original columns for debugging
    print(f"\nOriginal columns for {year}:")
    print(df.columns.tolist())
    
    # Create a mapping dictionary for each year's columns
    if year == 2023:
        column_mapping = {
            'estimated prevalence of modern slavery per 1,000 population': 'prevalence_per_1000',
            'estimated number of people in modern slavery': 'total_victims',
            'governance issues': 'governance_score',
            'lack of basic needs': 'basic_needs_score',
            'inequality': 'inequality_score',
            'disenfranchised groups': 'disenfranchised_score',
            'effects of conflict': 'conflict_score',
            'total vulnerability score (%)': 'vulnerability_total',
            'survivors of slavery are identified and supported to exit and remain out of modern slavery (%)': 'victim_support_score',
            'criminal justice mechanisms function effectively to prevent modern slavery (%)': 'justice_score',
            'coordination occurs at the national and regional level and across borders, and governments are held to account for their response (%)': 'coordination_score',
            'risk factors, such as attitudes, social systems, and institutions that enable modern slavery are addressed (%)': 'risk_address_score',
            'government and business stop sourcing goods and services produced by forced labour (%)': 'supply_chain_score',
            'government response total (%)': 'government_response_total',
            'female': 'lfpr_female',
            'male': 'lfpr_male',
            'total_lfpr': 'lfpr_total'
        }
    elif year == 2018:
        column_mapping = {
            'est. prevalence of population in modern slavery (victims per 1,000 population)': 'prevalence_per_1000',
            'est. number of people in modern slavery': 'total_victims',
            'factor one governance issues': 'governance_score',
            'factor two nourishment and access': 'basic_needs_score',
            'factor three inequality': 'inequality_score',
            'factor four disenfranchised groups': 'disenfranchised_score',
            'factor five effects of conflict': 'conflict_score',
            'final overall (normalised, weighted) vulnerability score': 'vulnerability_total',
            'support survivors': 'victim_support_score',
            'criminal justice': 'justice_score',
            'coordination': 'coordination_score',
            'address risk': 'risk_address_score',
            'supply chains': 'supply_chain_score',
            'total': 'government_response_total',
            'female': 'lfpr_female',
            'male': 'lfpr_male',
            'total_lfpr': 'lfpr_total'
        }
    elif year == 2016:
        column_mapping = {
            'estimated proportion of population in modern slavery': 'prevalence_per_1000',
            'estimated number in modern slavery': 'total_victims',
            'dimension 1: political rights and safety': 'governance_score',
            'dimension 2: financial and health protections': 'basic_needs_score',
            'dimension 3: protection for the most vulnerable': 'inequality_score',
            'dimension 4: conflict': 'conflict_score',
            'mean vulnerability': 'vulnerability_total',
            'milestone 1: victims are supported to exit slavery (%)': 'victim_support_score',
            'milestone 2: criminal justice responses (%)': 'justice_score',
            'milestone 3: coordination and accountability (%)': 'coordination_score',
            'milestone 4: addressing risk (%)': 'risk_address_score',
            'milestone 5: investigating supply chains (%)': 'supply_chain_score',
            'total score (/100)': 'government_response_total',
            'female': 'lfpr_female',
            'male': 'lfpr_male',
            'total': 'lfpr_total'
        }
    
    # Common columns across all years
    common_columns = {
        'country': 'country',
        'population': 'population',
        'region': 'region',
        'corruption': 'corruption_score',
        'democracy score': 'democracy_score',
        'gdp per capita': 'gdp_per_capita',
        'migration': 'migration_rate'
    }

    # Before standardization, check for missing columns
    missing_mappings = []
    for old_col in df.columns:
        if old_col not in column_mapping and old_col not in ['year']:
            missing_mappings.append(old_col)
            print(f"Warning: No mapping found for column: {old_col}")
    
    # Combine the year-specific mapping with common columns
    column_mapping.update(common_columns)

    # Create new DataFrame with standardized columns
    standardized_df = pd.DataFrame()
    
    # Copy over columns that exist in the mapping
    mapped_columns = []
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            standardized_df[new_col] = df[old_col]
            mapped_columns.append(new_col)
    
    # Ensure year column exists
    standardized_df['year'] = year
    
    # Print standardized columns for debugging
    print(f"\nStandardized columns for {year}:")
    print(standardized_df.columns.tolist())
    
    return standardized_df

def check_required_columns(df, year):
    """
    Check if all required columns are present in the dataset and have valid data
    """
    required_columns = {
        'country': str,
        'population': np.number,
        'prevalence_per_1000': np.number,
        'total_victims': np.number,
        'vulnerability_total': np.number,
        'governance_score': np.number,
        'basic_needs_score': np.number,
        'inequality_score': np.number,
        'conflict_score': np.number,
        'victim_support_score': np.number,
        'justice_score': np.number,
        'coordination_score': np.number,
        'risk_address_score': np.number,
        'supply_chain_score': np.number,
        'government_response_total': np.number
    }
    
    missing_columns = []
    invalid_data = []
    
    for col, dtype in required_columns.items():
        if col not in df.columns:
            missing_columns.append(col)
        else:
            # Check if numeric columns have valid data
            if dtype == np.number:
                if df[col].isnull().all():
                    invalid_data.append(f"{col} (all null)")
                elif df[col].dtype not in [np.float64, np.int64]:
                    invalid_data.append(f"{col} (wrong type: {df[col].dtype})")
    
    if missing_columns:
        print(f"\nWarning: Missing required columns for {year}:")
        for col in missing_columns:
            print(f"- {col}")
    
    if invalid_data:
        print(f"\nWarning: Invalid data in columns for {year}:")
        for issue in invalid_data:
            print(f"- {issue}")
    
    # Print summary statistics for numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    if numeric_cols:
        print(f"\nSummary statistics for {year}:")
        print(df[numeric_cols].describe())
    
    return len(missing_columns) == 0 and len(invalid_data) == 0

def standardize_numeric_values(df):
    """
    Standardize numeric columns using z-score standardization
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    # Identify numeric columns (excluding specific columns)
    exclude_cols = ['year', 'population']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create copy of dataframe
    std_df = df.copy()
    
    # Standardize each numeric column
    for col in numeric_cols:
        # Handle missing values
        if std_df[col].isnull().any():
            std_df[col] = std_df[col].fillna(std_df[col].median())
        
        # Reshape for StandardScaler
        values = std_df[col].values.reshape(-1, 1)
        
        # Standardize
        std_values = scaler.fit_transform(values)
        
        # Replace values in dataframe
        std_df[col] = std_values
    
    return std_df

def perform_multiple_testing_correction(p_values, method='bonferroni'):
    """
    Perform multiple testing correction on p-values
    """
    if method == 'bonferroni':
        # Bonferroni correction
        corrected_p = np.minimum(p_values * len(p_values), 1.0)
    else:
        # FDR correction
        _, corrected_p, _, _ = stats.multipletests(p_values, method='fdr_bh')
    
    return corrected_p

def calculate_correlations(df, target_col='prevalence_per_1000'):
    """
    Calculate correlations with target column and their p-values
    """
    correlations = {}
    p_values = {}
    
    # Get numeric columns (excluding specific columns)
    exclude_cols = ['year', 'population', 'total_victims']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col]
    
    for col in numeric_cols:
        try:
            # Get complete cases for both columns
            valid_data = df[[col, target_col]].dropna()
            
            if len(valid_data) > 1:  # Need at least 2 points for correlation
                x = valid_data[col]
                y = valid_data[target_col]
                
                # Check if either column is constant
                if x.std() != 0 and y.std() != 0:
                    corr, p_val = stats.pearsonr(x, y)
                    correlations[col] = corr
                    p_values[col] = p_val
        except Exception as e:
            print(f"Warning: Could not calculate correlation for {col}: {str(e)}")
            continue
    
    return correlations, p_values

def process_dataset(df, year):
    """
    Process a single dataset through the standardization pipeline
    """
    try:
        # Step 1: Standardize column names
        std_df = standardize_column_names(df, year)
        
        # Step 2: Check required columns and data validity
        if not check_required_columns(std_df, year):
            print(f"Warning: Data validation failed for {year}")
            # Continue processing but with a warning
        
        # Step 3: Handle missing values
        numeric_cols = std_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if std_df[col].isnull().any():
                median_val = std_df[col].median()
                std_df[col] = std_df[col].fillna(median_val)
                print(f"Filled {std_df[col].isnull().sum()} missing values in {col} with median")
        
        return std_df
    
    except Exception as e:
        print(f"Error processing dataset for year {year}: {str(e)}")
        return None
    
def validate_standardized_datasets(datasets):
    """
    Validate standardized datasets across years
    """
    print("\nValidating standardized datasets...")
    
    # Check for consistent columns across years
    all_columns = set()
    for year, df in datasets.items():
        all_columns.update(df.columns)
    
    print("\nChecking column consistency across years:")
    for col in sorted(all_columns):
        years_present = [year for year, df in datasets.items() if col in df.columns]
        if len(years_present) != len(datasets):
            print(f"Warning: Column '{col}' only present in years: {years_present}")
    
    # Check value ranges for key metrics
    key_metrics = ['prevalence_per_1000', 'vulnerability_total', 'government_response_total']
    print("\nChecking value ranges for key metrics:")
    for metric in key_metrics:
        print(f"\n{metric}:")
        for year, df in datasets.items():
            if metric in df.columns:
                print(f"{year}:")
                print(df[metric].describe())
    
    return True

def save_results(results, year):
    """
    Save standardized dataset and correlation results
    """
    try:
        # Save standardized dataset
        if 'data' in results and not results['data'].empty:
            results['data'].to_csv(f'processed_data/standardized_{year}.csv', index=False)
        
        # Create correlation results dataframe if correlations exist
        if results['correlations']:
            corr_df = pd.DataFrame({
                'variable': list(results['correlations'].keys()),
                'correlation': list(results['correlations'].values()),
                'p_value': list(results['p_values'].values()),
                'corrected_p_value': list(results['corrected_p_values'].values())
            })
            
            # Add significance indicator
            corr_df['significant'] = corr_df['corrected_p_value'] < 0.05
            
            # Sort by absolute correlation value
            corr_df['abs_correlation'] = abs(corr_df['correlation'])
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
            corr_df = corr_df.drop('abs_correlation', axis=1)
            
            # Save correlation results
            corr_df.to_csv(f'processed_data/correlations_{year}.csv', index=False)
            
            # Print summary of significant correlations
            sig_corrs = corr_df[corr_df['significant']]
            if not sig_corrs.empty:
                print(f"\nSignificant correlations for {year}:")
                for _, row in sig_corrs.iterrows():
                    print(f"{row['variable']}: r={row['correlation']:.3f}, p={row['corrected_p_value']:.3e}")
        
        return True
    
    except Exception as e:
        print(f"Error saving results for year {year}: {str(e)}")
        return False
    
def print_summary_statistics(df, year):
    """
    Print summary statistics for the dataset
    """
    print(f"\nSummary Statistics for {year}:")
    print("-" * 50)
    print(f"Number of observations: {len(df)}")
    print(f"Number of variables: {df.shape[1]}")
    print("\nNumeric variables summary:")
    print(df.select_dtypes(include=[np.number]).describe())

def main_standardize():
    """
    Main function to run the standardization pipeline
    """
    try:
        standardized_datasets = {}
        
        for year in [2016, 2018, 2023]:
            print(f"\nProcessing {year} dataset...")
            
            # Load dataset
            df = pd.read_csv(f'processed_data/filtered_merged_{year}.csv')
            
            # Process dataset
            std_df = process_dataset(df, year)
            
            if std_df is not None:
                standardized_datasets[year] = std_df
                
                # Save standardized dataset
                output_path = f'processed_data/standardized_{year}.csv'
                std_df.to_csv(output_path, index=False)
                print(f"Saved standardized dataset to {output_path}")
        
        # Validate standardized datasets
        if standardized_datasets:
            validate_standardized_datasets(standardized_datasets)
        
        return standardized_datasets
    
    except Exception as e:
        print(f"Error in main standardization process: {str(e)}")
        return None