import pandas as pd
import numpy as np

def standardize_column_names(df, year):
    """
    Standardize column names based on the year of the dataset
    """
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
    
    # Combine the year-specific mapping with common columns
    column_mapping.update(common_columns)
    
    # Create new DataFrame with standardized columns
    standardized_df = pd.DataFrame()
    
    # Copy over columns that exist in the mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            standardized_df[new_col] = df[old_col]
    
    # Ensure year column exists
    standardized_df['year'] = year
    
    return standardized_df

# [Rest of the code remains the same as in the previous artifact...]

