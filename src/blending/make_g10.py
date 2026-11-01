# g10: f1 85 + mehran TabPFN rank 15 (public-LB surface probe, no CV). Run from the competition folder.
import subprocess
import numpy as np, pandas as pd
from scipy.stats import rankdata

ids = pd.read_csv('data/sample_submission.csv').id.values
rk = lambda f: rankdata(pd.read_csv(f).set_index('id').loc[ids].iloc[:, -1].values) / len(ids)
r = 0.85 * rk('artifacts/submissions/b24/f1_mean5_213.csv') + 0.15 * rk('data/pub_subs/g24/mehran_tabpfn_submission.csv')
out = 'artifacts/submissions/b25/g10_f1_85_tab15.csv'
pd.DataFrame({'id': ids, 'satisfaction': r}).to_csv(out, index=False)
print('written', out)
subprocess.run(['kaggle', 'competitions', 'submit', 'playground-series-s6e10', '-f', out,
                '-m', 'rank blend g10_f1_85_tab15 (no CV; public-LB surface probe)'], check=True)
