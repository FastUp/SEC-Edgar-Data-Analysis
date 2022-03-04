#!/usr/bin/env python
# coding: utf-8

# # Getting Financial Data from SEC Edgar

# ## Automated processing using XBRL markup

# Automated analysis of regulatory filings has become much easier since the SEC introduced XBRL, a free, open, and global standard for the electronic representation and exchange of business reports. XBRL is based on XML; it relies on taxonomies that define the meaning of the elements of a report and map to tags that highlight the corresponding information in the electronic version of the report. One such taxonomy represents the US Generally Accepted Accounting Principles (GAAP).
# 
# The SEC introduced voluntary XBRL filings in 2005 in response to accounting scandals before requiring this format for all filers since 2009 and continues to expand the mandatory coverage to other regulatory filings. The SEC maintains a website that lists the current taxonomies that shape the content of different filings and can be used to extract specific items.
# 
# There are several avenues to track and access fundamental data reported to the SEC:
# 
# As part of the EDGAR Public Dissemination Service (PDS), electronic feeds of accepted filings are available for a fee.
# The SEC updates RSS feeds every 10 minutes, which list structured disclosure submissions.
# There are public index files for the retrieval of all filings through FTP for automated processing.
# The financial statement (and notes) datasets contain parsed XBRL data from all financial statements and the accompanying notes.
# The SEC also publishes log files containing the internet search traffic for EDGAR filings through SEC.gov, albeit with a six-month delay.
# 
# 

# ## Building a fundamental data time series

# The scope of the data in the Financial Statement and Notes datasets consists of numeric data extracted from the primary financial statements (Balance sheet, income statement, cash flows, changes in equity, and comprehensive income) and footnotes on those statements. The data is available as early as 2009.
# 
# The folder 03_sec_edgar contains the notebook edgar_xbrl to download and parse EDGAR data in XBRL format, and create fundamental metrics like the P/E ratio by combining financial statement and price data.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path
from datetime import date
import json
from io import BytesIO
from zipfile import ZipFile, BadZipFile
import requests
import streamlit as st


import pandas_datareader.data as web
import pandas as pd

from pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import h5py
from pandas import HDFStore

import requests, zipfile
import os

# In[3]:


sns.set_style('whitegrid')


# In[4]:


data_path = 'Volumes/Seagate Backup Plus Drive/Parth Patel_Mac/SEC Data/' # perhaps set to external harddrive to accomodate large amount of data

if not data_path :
    os.mkdir(data_path)


# The following code downloads and extracts all historical filings contained in the Financial Statement and Notes (FSN) datasets for the given range of quarters:

# Downloading 40GB of Data

# In[5]:


SEC_URL = 'https://www.sec.gov/files/dera/data/financial-statement-and-notes-data-sets/'


# In[6]:


today = pd.Timestamp(date.today())
this_year = today.year
this_quarter = today.quarter
past_years = range(2009, this_year)


# In[7]:


filing_periods = [(y, q) for y in past_years for q in range(1, 5)]
filing_periods.extend([(this_year, q) for q in range(1, this_quarter + 1)])


# In[8]:




for i, (yr, qtr) in enumerate(filing_periods, 1):
    print(f'{yr}-Q{qtr}', end=' ', flush=True)
    filing = f'{yr}q{qtr}_notes.zip'
    
    path = data_path /f'{yr} + {qtr}' / 'source'
    
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    
    response = requests.get(
        
        
        
        
        + filing).content
    
    try:

        with ZipFile(BytesIO(response)) as zip_file:
            for file in zip_file.namelist():
                local_file = path / file
                
                if local_file.exists():
                    continue
                with local_file.open('wb') as output:
                    for line in zip_file.open(file).readlines():
                        output.write(line)
                        
    except BadZipFile:
        print('got bad zip file')
        continue


# In[9]:


import dill as pickle
g = pickle.dumps(response)
pickle.loads(g)


# # Finding the list of Companies(Sectorwise, CIK, Locationwise)

# In[10]:


sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_constituents = pd.read_html(sp_url, header=0)[0]


# In[11]:


sp500_constituents.info()


# In[12]:


sp500_constituents.head()


# In[13]:


Tech_stocks = sp500_constituents[sp500_constituents['GICS Sector'] == 'Information Technology']


# In[14]:


Tech_stocks


# # Save to parquet¶
# 

# The data is fairly large and to enable faster access than the original text files permit, it is better to convert the text files to binary, columnar parquet format (see Section 'Efficient data storage with pandas' in chapter 2 for a performance comparison of various data-storage options compatible with pandas DataFrames):

# In[15]:


for f in data_path.glob('**/*.tsv'):
    file_name = f.stem  + '.parquet'
    path = Path(f.parents[1]) / 'parquet'
    if (path / file_name).exists():
        continue
    if not path.exists():
        path.mkdir(exist_ok=True)
    try:
        df = pd.read_csv(f, sep='\t', encoding='latin1', low_memory=False)
        print(df)
    except:
        print(f)
    df.to_parquet(path / file_name)


# In[16]:


sub = pd.read_parquet(data_path / '2020_4' / 'parquet' / 'sub.parquet')
sub.info()


# # Get ADOBE submission¶

# The submission dataset contains the unique identifiers required to retrieve the filings: the Central Index Key (CIK) and the Accession Number (adsh). The following shows some of the information about Adobe's 2020Q2 10-Q filing:

# In[17]:


cik = 1633917   #paypal corp
company = sub[sub.cik == cik].T.dropna().squeeze()
key_cols = ['name', 'adsh', 'cik', 'name', 'sic', 'countryba', 'stprba',
            'cityba', 'zipba', 'bas1', 'form', 'period', 'fy', 'fp', 'filed']
company.loc[key_cols]


# In[18]:


company_subs = pd.DataFrame()
for sub in data_path.glob('**/sub.parquet'):
    sub = pd.read_parquet(sub)
    company_sub = sub[(sub.cik.astype(int) == cik) & (sub.form.isin(['10-Q', '10-K']))]
    company_subs = pd.concat([company_subs, company_sub])


# In[19]:


company_subs.form.value_counts()


# In[20]:


company_nums = pd.DataFrame()
for num in data_path.glob('**/num.parquet'):
    num = pd.read_parquet(num).drop('dimh', axis=1)
    company_num = num[num.adsh.isin(company_subs.adsh)]
    print(len(company_num))
    company_nums = pd.concat([company_nums, company_num])
company_nums.ddate = pd.to_datetime(company_nums.ddate, format='%Y%m%d')   
company_nums.to_parquet(data_path / 'company_nums.parquet')


# In[21]:


g = pickle.dumps(company_nums)
pickle.loads(g)


# In[22]:


company_nums.info()


# # Create P/E Ratio from EPS and stock price data

# We can select a useful field, such as Earnings per Diluted Share (EPS), that we can combine with market data to calculate the popular Price/Earnings (P/E) valuation ratio.
# 
# 

# In[23]:


stock_split = 0
split_date = pd.to_datetime('20140604')
split_date


# In[24]:


# Filter by tag; keep only values measuring 1 quarter
eps = company_nums[(company_nums.tag == 'EarningsPerShareDiluted')
                & (company_nums.qtrs == 1)].drop('tag', axis=1)

# Keep only most recent data point from each filing
eps = eps.groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))

# Adjust earnings prior to stock split downward
eps.loc[eps.ddate < split_date,'value'] = eps.loc[eps.ddate < split_date, 'value'].div(7)
eps = eps[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
eps = eps.rolling(4,min_periods=4).sum().dropna()


# In[25]:


eps.plot(lw=2, figsize=(14, 6), title='Diluted Earnings per Share')
plt.xlabel('')
plt.savefig('diluted eps', dps=300);


# In[26]:


symbol = 'PYPL.US'
 

company_stock = (web.
              DataReader(symbol, 'quandl', start=eps.index.min(), 
                         end = eps.index.max(),
                         api_key = 'J8qrR58jz1huGcFxGuqh')
              .resample('D')
              .last()
             .loc['2009':eps.index.max()])
company_stock.info()


# In[27]:


pe = company_stock.AdjClose.to_frame('price').join(eps.to_frame('eps'))
pe = pe.fillna(method='ffill').dropna()
pe['P/E Ratio'] = pe.price.div(pe.eps)
pe['P/E Ratio'].plot(lw=2, figsize=(14, 6), title='TTM P/E Ratio');


# In[28]:


pe.info()


# In[29]:


axes = pe.plot(subplots=True, figsize=(16,8), legend=False, lw=2)
axes[0].set_title('Adj. Close Price')
axes[1].set_title('Diluted Earnings per Share')
axes[2].set_title('Trailing P/E Ratio')
plt.tight_layout();


# # Explore Additional Fields

# In[30]:


company_nums.tag.value_counts()


# In[31]:


type(company_nums.tag.value_counts())


# In[32]:


company_nums.tag.value_counts().to_csv() 


# In[33]:


outlier_cutoff = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]
for lag in lags:
    daily_prices = company_stock.AdjClose.to_frame('price')
    data[f'return_{lag}m'] = (daily_prices
                           .pct_change(lag)
                           .stack()
                           .pipe(lambda x: 
                                 x.clip(lower=x.quantile(outlier_cutoff),
                                        upper=x.quantile(1-outlier_cutoff)))
                           .add(1)
                           .pow(1/lag)
                           .sub(1)
                           )
data = data.swaplevel().dropna()
data.info()


# In[34]:


print(data)


# In[35]:


for lag in [2,3,6,9,12]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)
    data[f'momentum_3_12'] = data[f'return_12m'].sub(data.return_3m)

#momentum: the difference between 3rd and 12th month returns


# In[36]:


data[f'momentum_3_12']


# In[37]:


data['return_1m'].index.names  = ['date', 'return_1m']


# # Using Fama French Model to predict Prices

# In[38]:


factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 
              'famafrench', start='2000')[0].drop('RF', axis=1)


factor_data.index = factor_data.index.to_timestamp()
# print(factor_data.index)
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data = factor_data.join(data['return_1m']).sort_index()
T = 24
# betas = (factor_data
#          .groupby(level='ticker', group_keys=False)
#          .apply(lambda x: PandasRollingOLS(window=min(T, x.shape[0]-1), 
#                                            y=x.return_1m, x=x.drop('return_1m', axis=1)).beta))

betas = (factor_data
         .apply(lambda x: PandasRollingOLS(window=min(T, x.shape[0]-1), 
                                           y=x.return_1m, x=x.drop('return_1m', axis=1)).beta))


# In[39]:


print(betas)


# # Lagged Returns 

# In[40]:


for t in range(1, 10, 2 ):
    data[f'return_1m_t-{t}'] = data.return_1m.shift(t)
    print(data[f'return_1m_t-{t}'])


# In[41]:


# !brew install ta-lib

from pip._internal import main as install
install(["install","ta-lib"])



# In[42]:



dailyprices = np.ndarray(727, daily_prices['price'])
dailyprices


# In[ ]:


company_stock['AdjClose'] = company_stock.AdjClose.bfill(axis = 0)
company_stock['AdjVolume'] = company_stock.AdjVolume.bfill(axis = 0)


# In[ ]:


import numpy as np
import talib
from talib import RSI, BBANDS


up, mid, low = talib.BBANDS(company_stock['AdjClose'], timeperiod=5, nbdevup=1, nbdevdn=1, 
                      matype=0)
rsi = RSI(company_stock['AdjClose'], timeperiod=14)


# In[ ]:


std = company_stock['AdjClose'].rolling(window = 2).std()
sma = company_stock['AdjClose'].rolling(window = 2).mean()
upper_bb = sma + std * 2
lower_bb = sma - std * 2
    


# In[60]:


upper_bb


# In[ ]:


data = pd.DataFrame({'PYPL': company_stock['AdjClose'], 'BB Up': upper_bb, 'BB Mid': mid, 
                     'BB down': lower_bb, 'RSI': rsi})
fig, axes= plt.subplots(nrows=2, figsize=(25, 10))
data.drop('RSI', axis=1).plot(ax=axes[0], lw=1, title='Bollinger Bands')
data['RSI'].plot(ax=axes[1], lw=1, title='Relative Strength Index')
axes[1].axhline(70, lw=1, ls='--', c='k')
axes[1].axhline(30, lw=1, ls='--', c='k')


# In[56]:


get_ipython().system('brew link --overwrite python@2')

from pykalman import KalmanFilter
# from filterpy.kalman import KalmanFilter
kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)


# In[59]:


state_means, _ = kf.filter(company_stock['AdjClose'])


# In[58]:


price_smoothed = company_stock['AdjClose'].to_frame('close')
price_smoothed['Kalman Filter'] = state_means
for months in [1, 2, 3]:
    price_smoothed[f'MA ({months}m)'] = (company_stock['AdjClose'].rolling(window=months * 21)
                                         .mean())
    ax = price_smoothed.plot(title='Kalman Filter vs Moving Average',
                         figsize=(14, 6), lw=1, rot=0)


# In[ ]:


# # Filter by tag; keep only values measuring 1 quarter
# import ddate
# current_liabilities = company_nums[(company_nums.tag == 'LiabilitiesCurrent',['ddate', 'value'])]
# current_assets = company_nums[(company_nums.tag == 'AssetsCurrent',['ddate', 'value'])]
# current_ratio  = [(current_assets / current_liabilities)]
      

# Keep only most recent data point from each filing
#eps = eps.groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))


#eps = eps.rolling(4,min_periods=4).sum().dropna()


# In[ ]:





# In[ ]:





# In[ ]:


fields = ['EarningsPerShareDiluted',
          'PaymentsOfDividendsCommonStock',
          'WeightedAverageNumberOfDilutedSharesOutstanding',
          'OperatingIncomeLoss',
          'NetIncomeLoss',
          'GrossProfit']


# # Dividends per Share

# In[ ]:


dividends = (company_nums
             .loc[company_nums.tag == 'PaymentsOfDividendsCommonStock', ['ddate', 'value']]
             .groupby('ddate')
             .mean())
shares = (company_nums
          .loc[company_nums.tag == 'WeightedAverageNumberOfDilutedSharesOutstanding', ['ddate', 'value']]
          .drop_duplicates()
          .groupby('ddate')
          .mean())
df = dividends.div(shares) #.dropna()
ax = df.plot.bar(figsize=(14, 5), title='Dividends per Share', legend=False)
ax.xaxis.set_major_formatter(mticker.FixedFormatter(df.index.strftime('%Y-%m')))


# # Bonus: Textual Information¶

# In[ ]:


txt = pd.read_parquet(data_path / '2020_2' / 'parquet' /  'txt.parquet')


# In[ ]:


txt.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




