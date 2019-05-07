import os
import pandas as pd


# In[75]:


# In[15]:


# Marking Scheme = 
#  Power = 2,4,8 & -1,-2,-4 with break seq
#  Fibo = 2,3,5 & -1,-1,-2 with break seq
# All or None = (n marks)
Answers={
    'J':{
        'q1': ['3','03'],'q2': ['5','05','14'],'q3': ['05','5','23'],'q4': ['BONUS','B'],'q5': ['A'],'q6': ['C'],'q7': ['C'],
        'q8': ['4','04'],    'q9': ['BONUS'],    'q10': ['08','8'],'q11': ['C'],'q12': ['C'],'q13': ['B'],
        'q14': ['C'],'q15': ['B'],'q16': ['B'],'q17': ['D'],'q18': ['C'],'q19': ['15'],'q20': ['2','3','02','03']
    },
    'H':{
        'q1': ['C'],'q2': ['C'],'q3': ['C'],'q4': ['C'],'q5': ['B'],'q6': ['BONUS'],'q7': ['3','03'],
        'q8': ['9','09'],    'q9': ['4','04'],'q10': ['45'],'q11': ['12'],'q12': ['16'],'q13': ['C'],
        'q14': ['A'],'q15': ['B'],'q16': ['23'],'q17': ['61'],'q18': ['B'],
        'q19': ['ABC','ACB','BAC','BCA','CBA','CAB'],'q20': ['C'],
        'q21': 'B','q22': 'A','q23': 'D','q24': 'D','q25': 'B',
    }
}

# Fibo is across the sections - Q4,5,6,7,13,
Sections = {
    'J':{
        'Power1':{'ques':[1,2,3],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Fibo1':{'ques':[4,5,6,7],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'Power2':{'ques':[8,9,10],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Fibo2':{'ques':[11,12,13,14],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'allNone1':{'ques':[15,16],'marks':9},
        'allNone2':{'ques':[17,18],'marks':12},
        'allNone3':{'ques':[19,20],'marks':6},
    },

    'H' : {
        'allNone1':{'ques':[1],'marks':8},
        'Power1':{'ques':[2,3,4],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        # 'allNone2':{'ques':[5,6],'marks':12},
        'allNone2':{'ques':[5,6],'marks':6},
        'Fibo1':{'ques':[7,8,9,10,11],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'allNone3':{'ques':[12],'marks':8},
        'Power2':{'ques':[13,14,15],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Proxy1':{'ques':[16,17],'+marks':5,'-marks':3},
        'Power3':{'ques':[18,19,20],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'TechnoFin':{'ques':[21,22,23,24,25]},
    },
    'JK' : {
        'Power1':{'ques':[1,2,3],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Fibo1':{'ques':[4,5,6,7],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'Power2':{'ques':[8,9,10],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Fibo2':{'ques':[11,12,13,14],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'allNone1':{'ques':[15,16],'marks':9},
        'allNone2':{'ques':[17,18],'marks':12},
        'allNone3':{'ques':[19,20],'marks':6},

    },
    'HK' : {
        'allNone1':{'ques':[1],'marks':8},
        'Power1':{'ques':[2,3,4,5],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'allNone2':{'ques':[6],'marks':12},
        'Fibo1':{'ques':[7,8,9,10,11],'+seq':[2,3,5,8,13,21],'-seq':[1,1,2,3,5,8]},
        'allNone3':{'ques':[12],'marks':8},
        'Power2':{'ques':[13,14,15],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},
        'Proxy1':{'ques':[16,17],'+marks':5,'-marks':3},
        'Power3':{'ques':[18,19,20],'+seq':[2,4,8,16],'-seq':[1,2,4,8,16]},

    },
}

qNos={
    'J':['q'+str(i) for i in range(1,21)],
    'H':['q'+str(i) for i in range(1,26)]
}

# In[76]:

once = 0
def report(Status,streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks):
    global once
    if(not once):
        once = 1
        print('Question\tStatus \t Streak\tSection \tMarks_Update\tMarked:\tAnswer:')
        
    print('%s \t %s \t\t %s \t %s \t %s \t %s \t %s ' % (qNo,
          Status,str(streak), '['+scheme+'] ',(str(prevmarks)+' + '+str(currmarks)+' ='+str(marks)),str(marked),str(ans)))
# check sectionwise only.
def evaluate(resp,answers,sections,explain=False):
    marks,num_correct,num_wrong = 0,0,0
    allans = answers.items()
    sectionMarks={}
    prevmarks=0
    prevSectionMarks=0
    for scheme,section in sections.items():
        sectionques = section['ques']
        prevcorrect=None
        allflag=1
        streak=0
        for q in sectionques:
            qNo='q'+str(q)
            ans=answers[qNo]
            marked = resp.get(qNo, 'X')
            if(type(marked)==float or type(marked)==int):
            	marked=str(int(marked))
            
            firstQ = sectionques[0]
            lastQ = sectionques[len(sectionques)-1]
            unmarked = marked=='X' or marked==''
            bonus = 'BONUS' in ans
            correct = bonus or (marked in ans)
            inrange=0
            
# ('q13(Power2) Correct(streak0) -3 + 2 = -1', 'C', ['C'])
# ('q14(Power2) Correct(streak0) -1 + 2 = 1', 'A', ['A'])
# ('q15(Power2) Incorrect(streak0) 1 + -1 = 0', 'C', ['B'])
            if(unmarked or int(q)==firstQ):
                streak=0
            elif(prevcorrect == correct):
                streak+=1
            else:
                streak=0
                
    
            if( 'allNone' in scheme):
                #loop on all sectionques
                allflag = allflag and correct
                if(q == lastQ ):
                    #at the end check allflag
                    prevcorrect = correct
                    currmarks = section['marks'] if allflag else 0
                else:
                    currmarks = 0
                
            elif('Proxy' in scheme):
                a=int(ans[0])
                #proximity check
                inrange = 1 if unmarked else (float(abs(int(marked) - a))/float(a) <= 0.25)
                currmarks = section['+marks'] if correct else (0 if inrange else -section['-marks'])                
                
            elif('Fibo' in scheme or 'Power' in scheme):
                currmarks = section['+seq'][streak] if correct else (0 if unmarked else -section['-seq'][streak])                
            elif('TechnoFin' in scheme):
                currmarks = 0
            else:
                print('Invalid Sections')
            prevmarks=marks
            marks += currmarks
            
            if(explain):
                if bonus:
                    report('BonusQ',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif correct:
                    report('Correct',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif unmarked:
                    report('Unmarked',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                elif inrange:
                    report('InProximity',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
                else:
                    report('Incorrect',streak,scheme,qNo,marked,ans,prevmarks,currmarks,marks)
            if correct:
                num_correct+=1
            elif not unmarked:
                num_wrong+=1
            prevcorrect = correct
        
        sectionMarks[scheme]=marks-prevSectionMarks
        prevSectionMarks=marks
            
    return marks,num_correct,num_wrong,sectionMarks

import numpy as np


kv = 0

Qs=['q'+str(i) for i in range(1,21)]+['t'+str(i) for i in range(1,6)]
sheetCols=['batch','error','filename','path','roll']+Qs
unionSections=['Power1','Power2','Power3','Fibo1','Fibo2','allNone1','allNone2','allNone3','Proxy1','TechnoFin']
resultSheetCols=sheetCols+['score','num_correct','num_wrong']+unionSections

# y = pd.read_csv('results/multiMarkedSheet.csv')[['roll','score']]
# y = y.replace(np.nan,'',regex=True)
x = pd.read_csv('results/badRollSheet.csv')[sheetCols]
x = x.replace(np.nan,'',regex=True)


resultFileJ = 'results/JScoreWithNums'+('KVJNV' if kv else '')+'Results2017_test.csv'
resultFileH = 'results/HScoreWithNums'+('KVJNV' if kv else '')+'Results2017_test.csv'
if(not os.path.exists(resultFileJ)):
    with open(resultFileJ,'a') as f:
        results=[resultSheetCols]
        pd.DataFrame(results).to_csv(f,header=False)
else:
    print('WARNING : Appending to Previous Result file!')

if(not os.path.exists(resultFileH)):
    with open(resultFileH,'a') as f:
        results=[resultSheetCols]
        pd.DataFrame(results).to_csv(f,header=False)
else:
    print('WARNING : Appending to Previous Result file!')
intQs ={
'J' : [ 'q'+str(qNo) for qNo in (range(1,4)+range(8,11)+range(19,21))],
'H'	:[ 'q'+str(qNo) for qNo in (range(12,13)+range(16,18)+range(7,12))]
} 
# x[intQs[squad]]=x[intQs[squad]].astype(int)
explain=0
with open(resultFileJ,'a') as fJ:
    with open(resultFileH,'a') as fH:
        counterx,countery=0,0
        for i,row in enumerate(x.iterrows()):
            # results/allResults
            squad = row[1].roll[0]
            # print(squad)
            # debug=raw_input()
            score,nc,nw,sectionMarks = evaluate(dict(row[1]),Answers[squad],Sections[squad],explain=explain)
            f_ = fJ if squad=='J' else fH
            # print(sectionMarks.items())
            if(explain):
                debug=raw_input()
            secMarks=[]
            for x in range(len(unionSections)):
                try:
                    secMarks.append(sectionMarks[unionSections[x]])
                except:
                    secMarks.append('')
            pd.DataFrame(list(row[1])+[score,nc,nw]+secMarks).T.to_csv(f_,header=False)
            # y_i=y.iloc[i]
            # scorey = y_i.score
            # rolly = y_i.roll
            # if(scorey!='' and scorey!=0 and score!=int(scorey)):
            #     countery+=1
            #     print('Error: wrong scores : ',row[1]['roll'],score,rolly,scorey,countery)
            # else:
            #     counterx+=1
            #     print('Correct score',score,scorey,counterx)
            print(row[1]['roll'],score,nc,nw,sectionMarks)
