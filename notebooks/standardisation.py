import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import ConstantInputWarning
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

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

def convert_data_types(df):
        """
        Convert columns to appropriate data types and handle any formatting issues
        """
        try:
            # Dictionary of columns and their desired data types
            numeric_columns = {
                'population': 'float64',
                'total_victims': 'float64',
                'prevalence_per_1000': 'float64',
                'governance_score': 'float64',
                'basic_needs_score': 'float64',
                'inequality_score': 'float64',
                'disenfranchised_score': 'float64',
                'conflict_score': 'float64',
                'vulnerability_total': 'float64',
                'victim_support_score': 'float64',
                'justice_score': 'float64',
                'coordination_score': 'float64',
                'risk_address_score': 'float64',
                'supply_chain_score': 'float64',
                'government_response_total': 'float64',
                'corruption_score': 'float64',
                'democracy_score': 'float64',
                'gdp_per_capita': 'float64',
                'migration_rate': 'float64',
                'lfpr_female': 'float64',
                'lfpr_male': 'float64',
                'lfpr_total': 'float64'
            }
            
            for col, dtype in numeric_columns.items():
                if col in df.columns:
                    try:
                        # Remove any commas and convert to numeric
                        df[col] = df[col].astype(str).str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"Successfully converted {col} to numeric")
                    except Exception as e:
                        print(f"Warning: Could not convert {col} to {dtype}: {str(e)}")
            
            return df
        
        except Exception as e:
            print(f"Error in data type conversion: {str(e)}")
            return df
        
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

def perform_multiple_testing_correction(p_values, method='bonferroni', alpha=0.05):
        """
        Perform multiple testing correction on p-values
        
        Parameters:
        - p_values: List or array of p-values
        - method: Correction method ('bonferroni', 'holm', 'simes', 'fdr_bh')
        - alpha: Significance level
        
        Returns:
        - Corrected p-values
        """
        from statsmodels.stats.multitest import multipletests
        
        # Perform multiple testing correction
        reject, corrected_p, _, _ = multipletests(
            p_values, 
            alpha=alpha, 
            method=method
        )
        
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
            
            # Step 2: Convert data types
            std_df = convert_data_types(std_df)
            
            # Step 3: Check required columns and data validity
            if not check_required_columns(std_df, year):
                print(f"Warning: Data validation failed for {year}")
            
            # Step 4: Handle missing values
            numeric_cols = std_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if std_df[col].isnull().any():
                    median_val = std_df[col].median()
                    std_df[col] = std_df[col].fillna(median_val)
                    print(f"Filled {std_df[col].isnull().sum()} missing values in {col} with median")
            
            # Step 5: Standardize numeric values
            std_df = standardize_numeric_values(std_df)
            
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

def ensure_consistent_columns(datasets):
        """
        Ensure all datasets have the same columns
        """
        try:
            print("\nEnsuring consistent columns across datasets...")
            
            # Get all unique columns
            all_columns = set()
            for df in datasets.values():
                all_columns.update(df.columns)
            
            # Core columns that should always be present
            core_columns = [
                'country', 'year', 'prevalence_per_1000', 'total_victims',
                'vulnerability_total', 'governance_score', 'basic_needs_score',
                'inequality_score', 'conflict_score', 'government_response_total',
                'victim_support_score', 'justice_score', 'coordination_score',
                'risk_address_score', 'supply_chain_score', 'corruption_score',
                'democracy_score'
            ]
            
            # Add missing columns with NaN values
            for year, df in datasets.items():
                print(f"\nProcessing {year} dataset:")
                
                # Check core columns
                missing_core = [col for col in core_columns if col not in df.columns]
                if missing_core:
                    print(f"Warning: Missing core columns: {missing_core}")
                    for col in missing_core:
                        datasets[year][col] = np.nan
                
                # Add other columns that exist in any dataset
                missing_cols = [col for col in all_columns if col not in df.columns]
                if missing_cols:
                    print(f"Adding missing columns: {missing_cols}")
                    for col in missing_cols:
                        datasets[year][col] = np.nan
                
                # Ensure consistent column order
                datasets[year] = datasets[year].reindex(columns=sorted(all_columns))
            
            return datasets
        
        except Exception as e:
            print(f"Error ensuring consistent columns: {str(e)}")
            return datasets

def create_visualizations(datasets):
        """
        Create visualizations for key relationships
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Set style
            plt.style.use('default')
            sns.set_theme(style="whitegrid")
            
            # Key variables to plot
            variables = [
                'vulnerability_total', 'governance_score', 'corruption_score',
                'democracy_score', 'gdp_per_capita', 'government_response_total'
            ]
            
            # Create scatter plots
            for var in variables:
                fig = plt.figure(figsize=(15, 5))
                
                for i, (year, df) in enumerate(datasets.items(), 1):
                    if var in df.columns and 'prevalence_per_1000' in df.columns:
                        ax = fig.add_subplot(1, 3, i)
                        
                        # Create scatter plot
                        sns.scatterplot(data=df, x=var, y='prevalence_per_1000', ax=ax)
                        
                        # Add trend line
                        sns.regplot(data=df, x=var, y='prevalence_per_1000', 
                                scatter=False, color='red', ax=ax)
                        
                        # Calculate correlation
                        mask = df[[var, 'prevalence_per_1000']].notna().all(axis=1)
                        if mask.sum() > 1:
                            corr = stats.pearsonr(
                                df.loc[mask, var],
                                df.loc[mask, 'prevalence_per_1000']
                            )[0]
                            ax.set_title(f'{year}\nr = {corr:.3f}')
                        
                        ax.set_xlabel(var.replace('_', ' ').title())
                        ax.set_ylabel('Modern Slavery Prevalence')
                
                plt.tight_layout()
                plt.savefig(f'plots/{var}_relationship.png')
                plt.close()
            
            # Create correlation heatmaps
            for year, df in datasets.items():
                plt.figure(figsize=(12, 10))
                
                # Select numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                        fmt='.2f', square=True)
                
                plt.title(f'Correlation Heatmap - {year}')
                plt.tight_layout()
                plt.savefig(f'plots/correlation_heatmap_{year}.png')
                plt.close()
            
            print("\nVisualization files saved in 'plots' directory")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

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

def perform_regression_analysis(datasets):
        """
        Perform regression analysis for each year
        """
        try:
            print("\nPerforming Regression Analysis:")
            
            # Independent variables
            predictors = [
                'vulnerability_total', 'governance_score', 'corruption_score',
                'democracy_score', 'basic_needs_score', 'inequality_score',
                'conflict_score', 'government_response_total'
            ]
            
            for year, df in datasets.items():
                print(f"\nYear {year}:")
                
                # Get available predictors
                available_predictors = [var for var in predictors if var in df.columns]
                
                if 'prevalence_per_1000' not in df.columns:
                    print("Warning: No prevalence column found")
                    continue
                
                # Prepare data
                model_df = df[['prevalence_per_1000'] + available_predictors].copy()
                model_df = model_df.dropna()
                
                if len(model_df) < len(available_predictors) + 2:
                    print("Not enough complete cases for regression")
                    continue
                
                # Standardize predictors
                X = model_df[available_predictors]
                y = model_df['prevalence_per_1000']
                
                # Add constant
                X = sm.add_constant(X)
                
                # Fit model
                model = sm.OLS(y, X).fit()
                
                # Print results
                print("\nRegression Results:")
                print(f"R-squared: {model.rsquared:.4f}")
                print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
                print("\nCoefficients:")
                
                # Get coefficients and p-values
                coef_df = pd.DataFrame({
                    'coef': model.params,
                    'std_err': model.bse,
                    't_value': model.tvalues,
                    'p_value': model.pvalues
                })
                
                # Apply multiple testing correction
                coef_df['corrected_p'] = perform_multiple_testing_correction(
                    coef_df['p_value']
                )
                
                # Save results
                coef_df.to_csv(f'processed_data/regression_results_{year}.csv')
                
                # Print all coefficients
                for var in coef_df.index:
                    print(f"\n{var}:")
                    print(f"  Coefficient: {coef_df.loc[var, 'coef']:.3f}")
                    print(f"  t-value: {coef_df.loc[var, 't_value']:.3f}")
                    print(f"  Original p-value: {coef_df.loc[var, 'p_value']:.3e}")
                    print(f"  Corrected p-value: {coef_df.loc[var, 'corrected_p']:.3e}")
        
        except Exception as e:
            print(f"Error in regression analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())

def analyze_correlations_with_correction(datasets):
        """
        Analyze correlations with multiple testing correction
        """
        try:
            print("\nAnalyzing Correlations with Multiple Testing Correction:")
            
            key_variables = [
                'vulnerability_total', 'governance_score', 'corruption_score',
                'democracy_score', 'basic_needs_score', 'inequality_score',
                'conflict_score', 'government_response_total'
            ]
            
            for year, df in datasets.items():
                print(f"\nYear {year}:")
                
                # Get complete cases only
                analysis_vars = ['prevalence_per_1000'] + [var for var in key_variables if var in df.columns]
                analysis_df = df[analysis_vars].dropna()
                
                if len(analysis_df) < 2:
                    print("Not enough complete cases for analysis")
                    continue
                
                # Calculate correlations
                correlations = {}
                p_values = {}
                
                for var in key_variables:
                    if var in analysis_df.columns:
                        mask = analysis_df[[var, 'prevalence_per_1000']].notna().all(axis=1)
                        if mask.sum() > 1:
                            x = analysis_df.loc[mask, var]
                            y = analysis_df.loc[mask, 'prevalence_per_1000']
                            corr, p_val = stats.pearsonr(x, y)
                            correlations[var] = corr
                            p_values[var] = p_val
                
                if correlations:
                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'variable': list(correlations.keys()),
                        'correlation': list(correlations.values()),
                        'p_value': list(p_values.values())
                    })
                    
                    # Perform multiple testing correction
                    results_df['corrected_p'] = perform_multiple_testing_correction(
                        results_df['p_value'].values
                    )
                    
                    # Sort by absolute correlation
                    results_df['abs_corr'] = abs(results_df['correlation'])
                    results_df = results_df.sort_values('abs_corr', ascending=False)
                    results_df = results_df.drop('abs_corr', axis=1)
                    
                    # Save results
                    results_df.to_csv(f'processed_data/correlation_results_{year}.csv')
                    
                    # Print significant correlations
                    print("\nSignificant correlations (corrected p < 0.05):")
                    sig_results = results_df[results_df['corrected_p'] < 0.05]
                    if not sig_results.empty:
                        for _, row in sig_results.iterrows():
                            print(f"{row['variable']}:")
                            print(f"  r = {row['correlation']:.3f}")
                            print(f"  Original p-value: {row['p_value']:.3e}")
                            print(f"  Corrected p-value: {row['corrected_p']:.3e}")
                    else:
                        print("No significant correlations found")
                else:
                    print("No valid correlations could be calculated")
        
        except Exception as e:
            print(f"Error in correlation analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())

def analyze_lfpr_correlations(datasets):
        """
        Analyze correlations between LFPR and modern slavery prevalence
        """
        print("\nAnalyzing LFPR Correlations:")
        
        lfpr_variables = [
            'lfpr_total', 'lfpr_female', 'lfpr_male',
            'prevalence_per_1000'
        ]
        
        control_variables = [
            'gdp_per_capita', 'corruption_score', 'democracy_score',
            'vulnerability_total', 'government_response_total'
        ]
        
        for year, df in datasets.items():
            print(f"\nYear {year}:")
            
            # Analyze LFPR correlations
            lfpr_results = {}
            for var in lfpr_variables[:-1]:  # Exclude prevalence
                if var in df.columns:
                    mask = df[[var, 'prevalence_per_1000']].notna().all(axis=1)
                    if mask.sum() > 1:
                        x = df.loc[mask, var]
                        y = df.loc[mask, 'prevalence_per_1000']
                        corr, p_val = stats.pearsonr(x, y)
                        lfpr_results[var] = {
                            'correlation': corr,
                            'p_value': p_val
                        }
            
            # Print LFPR correlations
            print("\nLFPR Correlations with Modern Slavery Prevalence:")
            for var, results in lfpr_results.items():
                print(f"\n{var}:")
                print(f"  r = {results['correlation']:.3f}")
                print(f"  p-value = {results['p_value']:.3e}")
            
            # Analyze by gender gap
            if 'lfpr_female' in df.columns and 'lfpr_male' in df.columns:
                df['lfpr_gender_gap'] = df['lfpr_male'] - df['lfpr_female']
                mask = df[['lfpr_gender_gap', 'prevalence_per_1000']].notna().all(axis=1)
                if mask.sum() > 1:
                    corr, p_val = stats.pearsonr(
                        df.loc[mask, 'lfpr_gender_gap'],
                        df.loc[mask, 'prevalence_per_1000']
                    )
                    print("\nLFPR Gender Gap:")
                    print(f"  r = {corr:.3f}")
                    print(f"  p-value = {p_val:.3e}")
            
            # Control variable correlations
            print("\nControl Variable Correlations:")
            for var in control_variables:
                if var in df.columns:
                    mask = df[[var, 'prevalence_per_1000']].notna().all(axis=1)
                    if mask.sum() > 1:
                        corr, p_val = stats.pearsonr(
                            df.loc[mask, var],
                            df.loc[mask, 'prevalence_per_1000']
                        )
                        print(f"\n{var}:")
                        print(f"  r = {corr:.3f}")
                        print(f"  p-value = {p_val:.3e}")

def perform_lfpr_regression(datasets):
        """
        Perform regression analysis focusing on LFPR
        """
        print("\nPerforming LFPR Regression Analysis:")
        
        for year, df in datasets.items():
            print(f"\nYear {year}:")
            
            # Prepare variables
            predictors = [
                'lfpr_total', 'lfpr_gender_gap',
                'gdp_per_capita', 'corruption_score',
                'vulnerability_total', 'government_response_total'
            ]
            
            # Calculate gender gap if possible
            if 'lfpr_female' in df.columns and 'lfpr_male' in df.columns:
                df['lfpr_gender_gap'] = df['lfpr_male'] - df['lfpr_female']
            
            # Get available predictors
            available_predictors = [var for var in predictors if var in df.columns]
            
            if 'prevalence_per_1000' not in df.columns:
                print("Warning: No prevalence column found")
                continue
            
            # Prepare data
            model_df = df[['prevalence_per_1000'] + available_predictors].copy()
            model_df = model_df.dropna()
            
            if len(model_df) < len(available_predictors) + 2:
                print("Not enough complete cases for regression")
                continue
            
            # Standardize predictors
            X = model_df[available_predictors]
            y = model_df['prevalence_per_1000']
            
            # Add constant
            X = sm.add_constant(X)
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Print results
            print("\nRegression Results:")
            print(f"R-squared: {model.rsquared:.4f}")
            print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            
            # Print coefficients
            print("\nCoefficients:")
            for var in model.params.index:
                print(f"\n{var}:")
                print(f"  Coefficient: {model.params[var]:.3f}")
                print(f"  t-value: {model.tvalues[var]:.3f}")
                print(f"  p-value: {model.pvalues[var]:.3e}")

def create_advanced_visualizations(datasets):
        """
        Create advanced visualizations for correlation and regression analysis
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create plots directory
        os.makedirs('plots/advanced', exist_ok=True)
        
        # Correlation Heatmaps
        for year, df in datasets.items():
            plt.figure(figsize=(12, 10))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                        center=0, fmt='.2f', square=True)
            plt.title(f'Correlation Heatmap - {year}')
            plt.tight_layout()
            plt.savefig(f'plots/advanced/correlation_heatmap_{year}.png')
            plt.close()
        
        # Scatter plots with regression lines
        key_variables = [
            'vulnerability_total', 'governance_score', 
            'corruption_score', 'democracy_score'
        ]
        
        for var in key_variables:
            plt.figure(figsize=(15, 5))
            
            for i, (year, df) in enumerate(datasets.items(), 1):
                plt.subplot(1, 3, i)
                sns.regplot(
                    x=var, 
                    y='prevalence_per_1000', 
                    data=df, 
                    scatter_kws={'alpha':0.5},
                    line_kws={'color': 'red'}
                )
                plt.title(f'{year} - {var} vs Modern Slavery')
            
            plt.tight_layout()
            plt.savefig(f'plots/advanced/{var}_regression.png')
            plt.close()

def stratified_analysis(datasets):
        """
        Perform stratified analysis by region or income group
        """
        for year, df in datasets.items():
            # Group by region
            if 'region' in df.columns:
                regional_analysis = df.groupby('region')['prevalence_per_1000'].agg(['mean', 'median', 'std'])
                print(f"\nRegional Analysis for {year}:")
                print(regional_analysis)
            
            # Optional: Categorize by GDP or other metrics
            df['income_group'] = pd.qcut(df['gdp_per_capita'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            income_analysis = df.groupby('income_group')['prevalence_per_1000'].agg(['mean', 'median', 'std'])
            print(f"\nIncome Group Analysis for {year}:")
            print(income_analysis)

def robustness_checks(datasets):
        """
        Perform various robustness checks
        """
        from sklearn.linear_model import HuberRegressor
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        def calculate_vif(X):
            """Calculate Variance Inflation Factor"""
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            return vif_data
        
        for year, df in datasets.items():
            print(f"\nRobustness Checks for {year}:")
            
            # Select predictors
            predictors = ['vulnerability_total', 'governance_score', 'corruption_score']
            X = df[predictors]
            y = df['prevalence_per_1000']
            
            # Multicollinearity Check
            print("\nVariance Inflation Factors:")
            print(calculate_vif(X))
            
            # Robust Regression
            robust_reg = HuberRegressor()
            robust_reg.fit(X, y)
            
            print("\nRobust Regression Coefficients:")
            for name, coef in zip(predictors, robust_reg.coef_):
                print(f"{name}: {coef}")


def apply_multiple_testing_corrections(correlations_dict):
        """
        Apply multiple testing corrections to correlation p-values
        """
        from statsmodels.stats.multitest import multipletests
        
        # Extract p-values
        p_values = list(correlations_dict['p_values'].values())
        
        # Apply different correction methods
        corrections = {
            'bonferroni': multipletests(p_values, method='bonferroni')[1],
            'holm': multipletests(p_values, method='holm')[1],
            'fdr_bh': multipletests(p_values, method='fdr_bh')[1]
        }
        
        # Add corrected p-values to results
        results = correlations_dict.copy()
        for method, corrected_p in corrections.items():
            results[f'{method}_corrected_p'] = dict(zip(
                correlations_dict['p_values'].keys(),
                corrected_p
            ))
        
        return results

def create_correlation_plots(datasets):
        """
        Create correlation plots using matplotlib
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create output directory
        os.makedirs('plots', exist_ok=True)
        
        # Key variables to plot
        variables = ['lfpr_total', 'lfpr_female', 'lfpr_male', 'gdp_per_capita',
                    'corruption_score', 'vulnerability_total']
        
        for var in variables:
            plt.figure(figsize=(15, 5))
            
            for i, (year, df) in enumerate(datasets.items(), 1):
                if var in df.columns:
                    plt.subplot(1, 3, i)
                    sns.scatterplot(data=df, x=var, y='prevalence_per_1000')
                    sns.regplot(data=df, x=var, y='prevalence_per_1000', 
                            scatter=False, color='red')
                    plt.title(f'{year}')
                    plt.xlabel(var.replace('_', ' ').title())
                    plt.ylabel('Modern Slavery Prevalence')
            
            plt.tight_layout()
            plt.savefig(f'plots/{var}_relationship.png')
            plt.close()

def perform_stratified_analysis(datasets):
        """
        Perform stratified analysis by region and income level
        """
        results = {}
        
        for year, df in datasets.items():
            results[year] = {}
            
            # Regional analysis
            if 'region' in df.columns:
                regional_stats = df.groupby('region')['prevalence_per_1000'].agg([
                    'mean', 'median', 'std', 'count'
                ])
                results[year]['regional'] = regional_stats
            
            # Income level analysis (using GDP quartiles)
            if 'gdp_per_capita' in df.columns:
                df['income_group'] = pd.qcut(df['gdp_per_capita'], 
                                        q=4, 
                                        labels=['Low', 'Lower-Mid', 
                                                'Upper-Mid', 'High'])
                income_stats = df.groupby('income_group')['prevalence_per_1000'].agg([
                    'mean', 'median', 'std', 'count'
                ])
                results[year]['income'] = income_stats
        
        return results


def main_standardize():
        """Main standardization function"""
        try:
            standardized_datasets = {}
            
            for year in [2016, 2018, 2023]:
                print(f"\nProcessing {year} dataset...")
                df = pd.read_csv(f'processed_data/filtered_merged_{year}.csv')
                std_df = process_dataset(df, year)
                
                if std_df is not None:
                    standardized_datasets[year] = std_df
            
            # Ensure consistent columns
            standardized_datasets = ensure_consistent_columns(standardized_datasets)
            
            # Perform LFPR-specific analyses
            analyze_lfpr_correlations(standardized_datasets)
            perform_lfpr_regression(standardized_datasets)
            
            # Validate
            validate_standardized_datasets(standardized_datasets)
            
            # Add these lines
            create_advanced_visualizations(standardized_datasets)
            stratified_analysis(standardized_datasets)
            robustness_checks(standardized_datasets)
            
            return standardized_datasets
        except Exception as e:
            print(f"Error in main standardization process: {str(e)}")
            return None