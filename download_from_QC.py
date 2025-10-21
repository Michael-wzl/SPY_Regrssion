import pandas as pd
from datetime import datetime
from QuantConnect.Research import QuantBook
from QuantConnect import Resolution

qb = QuantBook()

# Whether to include extended hours data
EXTENDED_HOURS = True

# Add SPY minute data
spy = qb.AddEquity("SPY", Resolution.Minute, extendedMarketHours=EXTENDED_HOURS).Symbol

start = pd.Timestamp("2014-12-30 04:00", tz="America/New_York")
end   = pd.Timestamp("2025-06-03 20:00", tz="America/New_York")

hist = qb.History(spy, start, end, Resolution.DAILY)

if isinstance(hist.index, pd.MultiIndex):
    df = hist.loc[spy].reset_index()
else:
    df = hist.reset_index()

# Convert time to UTC if needed
if 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'], utc=True)#.dt.tz_convert('America/New_York')

#mask = (df['time'].dt.time >= time(9, 0)) & (df['time'].dt.time <= time(10, 0))
#df = df.loc[mask, ['time', 'open', 'high', 'low', 'close', 'volume']].copy()

# Print CSV text to output
csv_text = df.to_csv(index=False)
print(csv_text)