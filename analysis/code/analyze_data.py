import pandas as pd
import numpy as np
from linearmodels import PanelOLS

### DEFINE
def main():
    df, a = import_data()
    df = import_data()
    fit = run_regression(df)
    formatted = format_model(fit)

    fit_a = run_regression(a)
    formatted_a = format_model(fit_a)

    
    with open('output/regression.csv', 'w') as f:
        f.write('<tab:regression>' + '\n')
        formatted.to_csv(f, sep = '\t', index = False, header = False)
        f.write('<tab:regression>' + '\n')
        formatted_a.to_csv(f, sep = '\t', index = False, header = False)    

def import_data():
    df = pd.read_csv('input/data_cleaned.csv')
    df['post_tv'] = df['year'] > df['year_tv_introduced']
    a = df.loc[df['year'] >= 1960]
    
    return(df, a)
    
def run_regression(df):
    df = df.set_index(['county_id', 'year'])
    model = PanelOLS.from_formula('chips_sold ~ 1 + post_tv + EntityEffects + TimeEffects', data = df)
    fit = model.fit()
    
    return(fit)
    
def format_model(fit):
    formatted = pd.DataFrame({'coef'     : fit.params, 
                              'std_error': fit.std_errors, 
                              'p_value'  : fit.pvalues})
    formatted = formatted.loc[['post_tv']]
    
    return(formatted)
    
### EXECUTE
main()