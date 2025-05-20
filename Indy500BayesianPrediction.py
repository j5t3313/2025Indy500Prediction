
import re
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
from sklearn.preprocessing import StandardScaler


def parse_time_to_seconds(val):
    """Parse a time string or numeric to total seconds."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if re.match(r'^\d+(\.\d+)?$', s):
        return float(s)
    parts = s.split(':')
    nums = [float(p) for p in parts]
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    elif len(nums) == 1:
        return nums[0]
    else:
        raise ValueError(f"Unexpected time parts: {s}")
    return h * 3600 + m * 60 + sec


def load_preprocess_quali(path):
    df = pd.read_excel(path, dtype=str)
    df[['Year','Pos','No']] = df[['Year','Pos','No']].astype(int)
    time_cols = ['Lap 1 Time','Lap 2 Time','Lap 3 Time','Lap 4 Time','Total Time']
    for col in time_cols:
        df[col] = df[col].apply(parse_time_to_seconds)
    df['Average Speed'] = df['Average Speed'].astype(float)
    idx = df.groupby(['Year','Driver','No'])['Total Time'].idxmin()
    best = df.loc[idx].copy()
    best['AvgLapTime'] = best[['Lap 1 Time','Lap 2 Time','Lap 3 Time','Lap 4 Time']].mean(axis=1)
    best = best.rename(columns={'Pos':'QualiPos','Total Time':'TotalTime','Average Speed':'QualiAvgSpeed'})
    return best[['Year','No','Driver','QualiPos','TotalTime','AvgLapTime','QualiAvgSpeed']]


def load_preprocess_race(path):
    df = pd.read_excel(path, dtype=str)
    df[['Year','Pos','Start Pos','No','Laps','Pit Stops','Points']] = \
        df[['Year','Pos','Start Pos','No','Laps','Pit Stops','Points']].astype(int)
    df['ElapsedTime'] = df['Elapsed Time'].apply(parse_time_to_seconds)
    df['RaceAvgSpeed'] = df['Average Speed'].astype(float)
    df['Winner'] = (df['Pos'] == 1).astype(int)
    return df[['Year','No','Driver','Start Pos','Winner']]


def merge_and_engineer_features(quali, race):
    # Merge qualifying and race
    data = pd.merge(quali, race, on=['Year','No','Driver'], how='inner')
    # Experience: number of Indy 500s from 2020-2024
    exp = (race[race['Year'] <= 2024]
           .groupby('Driver')['Year']
           .nunique()
           .rename('Experience')
           .reset_index())
    data = data.merge(exp, on='Driver', how='left')
    data['Experience'] = data['Experience'].fillna(0).astype(int)
    # Recency weight
    data['RecencyWeight'] = 1 / (2025 - data['Year'] + 1)
    # Recency-weighted Quali position
    data['QualiPosRecency'] = data['QualiPos'] * data['RecencyWeight']
    return data, exp.set_index('Driver')['Experience']


def standardize(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def model_logistic(X, y=None):
    p = X.shape[1]
    intercept = numpyro.sample('intercept', dist.Normal(0, 10))
    coefs = numpyro.sample('coefs', dist.Normal(jnp.zeros(p), 10))
    logits = intercept + jnp.dot(X, coefs)
    numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)


def train_mcmc(X, y, rng_key):
    kernel = NUTS(model_logistic, init_strategy=init_to_median())
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=2)
    mcmc.run(rng_key, X, y)
    return mcmc.get_samples()


def predict_prob(trace, scaler, X_new, drivers):
    intercept = jnp.mean(trace['intercept'])
    coefs = jnp.mean(trace['coefs'], axis=0)
    Xs = scaler.transform(X_new)
    logits = intercept + jnp.dot(Xs, coefs)
    try:
        from jax.nn import sigmoid
        probs = sigmoid(logits)
    except ImportError:
        probs = 1 / (1 + np.exp(-logits))
    df = pd.DataFrame({'Driver': drivers, 'WinProb': np.array(probs)})
    df['WinProb (%)'] = (df['WinProb'] * 100).round(2)
    return df[['Driver','WinProb (%)']].sort_values('WinProb (%)', ascending=False)


def main():
    quali = load_preprocess_quali("\Indy500QualiALLYEARS.xlsx")
    race  = load_preprocess_race("\Indy500RaceALLYEARS.xlsx")
    data, exp_map = merge_and_engineer_features(quali, race)

    # Prepare training data
    train = data[data['Year'] <= 2024]
    features = ['QualiPos','TotalTime','AvgLapTime','QualiAvgSpeed','Experience','QualiPosRecency']
    X_train = train[features].values
    y_train = train['Winner'].values

    # Standardize & train
    Xs, scaler = standardize(X_train)
    rng_key = random.PRNGKey(0)
    trace = train_mcmc(Xs, y_train, rng_key)

    # Prepare 2025 test set
    test = load_preprocess_quali("Indy500QualiALLYEARS.xlsx")
    test = test[test['Year'] == 2025].copy()
    test['Experience'] = test['Driver'].map(exp_map).fillna(0).astype(int)
    test['RecencyWeight'] = 1 / (2025 - test['Year'] + 1)
    test['QualiPosRecency'] = test['QualiPos'] * test['RecencyWeight']
    X_test = test[features].values
    drivers = test['Driver'].values

    # Predict and print top 3
    results = predict_prob(trace, scaler, X_test, drivers)
    top3 = results.head(3)
    print("🏁 Top 3 predicted winners of the 2025 Indy 500 🏁:")
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        print(f"{rank}. {row['Driver']} ({row['WinProb (%)']}%)")

if __name__=='__main__':
    main()
