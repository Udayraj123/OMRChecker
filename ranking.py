import os
import pandas as pd
import numpy as np

squad = 'J'
kv = 0
explain=0

Qs=['q'+str(i) for i in range(1,21)]+['t'+str(i) for i in range(1,6)]
sheetCols=['batch','error','filename','path','roll']+Qs
resultSheetCols=sheetCols+['score','num_correct','num_wrong']
x = pd.read_csv('results/HScoreWithNumsResults2017.csv')[resultSheetCols]
x = x.replace(np.nan,'',regex=True)

manualResults = resultSheetCols
resultFile = 'results/reconfirmedTechno'+('KVJNV' if kv else '')+'Results2017.csv'
if(not os.path.exists(resultFile)):
    with open(resultFile,'a') as f:
        results=[resultSheetCols]
        pd.DataFrame(results).to_csv(f,header=False)
else:
    print('WARNING : Appending to Previous Result file!')

# x[intQs[squad]]=x[intQs[squad]].astype(int)   
x = x.sort_values(by=['score','num_correct','num_wrong'],ascending=False)
x.to_sql('test.sql','SQLAlchemy')
# with open(resultFile,'a') as f:
# 	for row in x.iterrows():
# 		print(list(row[1])[-3:])
