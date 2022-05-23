import os
import torch
import pandas as pd
from tqdm import tqdm

DATA_PATH = '../data/'
data = pd.read_csv(os.path.join(DATA_PATH,'sp500_stocks.csv'), index_col=[0], parse_dates=True) # Read csv file
companies = data['Symbol'].unique() # Get all the unique companies

tss, comp = [], []
for company in tqdm(companies, total=len(companies)):
    ts = data[data['Symbol'] == company]['Close'] # Get the price after the market closing
    ts.fillna(value=None, method='backfill', downcast=None, inplace=True) # Fill missing values
    resampled = ts.resample('W').mean() # Resample (downsample) ts to have accurate and regular measurements
    resampled = resampled.round(4) # Round floating point numbers
    hyp_dr_resample = pd.date_range(resampled.index.min(), resampled.index.max(), freq='W') # This would the optimal frequency (no missing days)
    if len(hyp_dr_resample) != len(resampled):
        print('Problem for {company}')
        print(f'Hypothesized entries after resampling: {hyp_dr_resample.shape[0]}, \
            Real entries  after resampling: {resampled.shape[0]}')
        continue

    tss.append(ts)
    comp.append(company)

# Save list to file for dataloading
torch.save(tss, os.path.join(DATA_PATH,'sp500_stocks_clean.pth'))
torch.save(comp, os.path.join(DATA_PATH,'sp500_stocks_clean_companies.pth'))