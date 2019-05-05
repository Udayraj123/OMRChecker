from constants import *

def merge_dicts(a,b):
    z=a.copy()
    z.update(b)
    return z

def calcGaps(PointsX,PointsY,numsX,numsY):
    gapsX = ( abs(PointsX[0]-PointsX[1])/(numsX[0]-1),abs(PointsX[2]-PointsX[3]) )
    gapsY = ( abs(PointsY[0]-PointsY[1])/(numsY[0]-1),abs(PointsY[2]-PointsY[3]) )
    return (gapsX,gapsY)

def maketemplateINT(qNos,start,numsX,numsY,gapsX,gapsY):
#     generate the coordinates
    qtags = {}
    values = {}
    templateINT=[]
    posx=start[0]
    qIndex = 0
    val = 0 
    for x in range(numsX[1]):#0,1,
        posy=start[1]
        for y in range(numsY[1]):#0,1,2,3,..9
            point = (posx,posy)
            templateINT.append(point)
            
            qtags[point]  = qNos[qIndex]
            values[point] = val
            
            val+=1
            
            posy+= (gapsY[1] if ((y+1) % numsY[0]==0) else gapsY[0])
        
        if ((x+1) % numsX[0]==0):
            val=0
            qIndex+=1
        
        posx+= (gapsX[1] if ((x+1) % numsX[0]==0) else gapsX[0])
    return templateINT,qtags,values
# q12H,qtags,values  = maketemplateINT(qNos=list(range(12,13)),start=start12H,numsX=(2,2),numsY=(1,10),gapsX=gapsXintH,gapsY=gapsYintH)

# In[73]:

def maketemplateMCQ(qNos,start,numsX,numsY,gapsX,gapsY):
#     generate the coordinates
    qtags = {}
    values = {}
    templateMCQ=[]
    posy=start[1]
    qIndex = 0
#     start from y
    for y in range(numsY[1]):#no of rows
        posx=start[0]
        for x in range(numsX[1]): #0,1,2,3 - no of options
            point = (posx,posy)
            templateMCQ.append(point)
            qtags[point]  = qNos[y] #here qNos is used
            values[point] = x
            posx+= (gapsX[1] if ((x+1) % numsX[0]==0) else gapsX[0])
        
        posy+= (gapsY[1] if ((y+1) % numsY[0]==0) else gapsY[0])
        
    return templateMCQ,qtags,values

def scalePts(pts,fac=1.2):
    spts=[]
    for i,pt in enumerate(pts):
        spts.append((pt[0]*fac,pt[1]*fac))
    return tuple(spts)


# Config for Manual fit - 
scalefac = 2
startRoll=(112,188) if kv else (113,184) 
endRoll = (478,473)
gapsYintJRoll=(20,31)
gapsXintJRoll=(39,39) 

start1to4J = (80,580)
start10to13J = (80,805)
start5to9J = (606,184)
start14to16J = (605,575)
start17to20J = (605,800)


gapsXintJ=(20,38)
gapsXintJ20=(20,42) if kv else (20,37)
gapsXmcqJ=(20,12)
gapsYmcqJ=(18,32)
gapsXintJ,gapsXintJ20,gapsXmcqJ ,gapsYmcqJ  = scalePts((gapsXintJ,gapsXintJ20,gapsXmcqJ ,gapsYmcqJ ),scalefac)
gapsYintJ=(10,31)



# startRoll,start1to3J,start4to7J,start11to18J,start8to10J,start19to20J,gapsXintJ,gapsXintJ20,gapsXmcqJ,gapsYmcqJ,gapsYintJ = scalePts([startRoll,start1to3J,start4to7J,start11to18J,start8to10J,start19to20J,gapsXintJ,gapsXintJ20,gapsXmcqJ,gapsYmcqJ,gapsYintJ],scalefac)
QTAGS={'J':{},'H':{}}
VALUES={'J':{},'H':{}}
# JUNIORS TEMPLATE
squad='J'
#COMMON ROLL TEMPLATE

roll_med,qtags,values = maketemplateINT(qNos=['Medium'],start=startRoll,numsX=(1,1),numsY=(1,2),gapsX=gapsXintJRoll,gapsY=gapsYintJRoll)
QTAGS[squad], VALUES[squad]  = merge_dicts(QTAGS[squad],qtags), merge_dicts(VALUES[squad],values)
roll_end,qtags,values = maketemplateINT(qNos=[ 'Roll'+str(i) for i in range(9) ],start=(startRoll[0]+18*scalefac,startRoll[1]),numsX=(1,9),numsY=(1,10),gapsX=gapsXintJRoll,gapsY=gapsYintJRoll)
QTAGS[squad], VALUES[squad]  = merge_dicts(QTAGS[squad],qtags), merge_dicts(VALUES[squad],values)
roll = roll_med+roll_end

q1to4J,qtags,values = maketemplateMCQ(qNos=list(range(1,5)),start=start1to4J,numsX=(4,4),numsY=(4,4),gapsX=gapsXmcqJ,gapsY=gapsYmcqJ)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q5to9J,qtags,values = maketemplateINT(qNos=list(range(5,10)),start=start5to9J,numsX=(2,10),numsY=(1,10),gapsX=gapsXintJ20,gapsY=gapsYintJ)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q10to13J,qtags,values  = maketemplateMCQ(qNos=list(range(10,14)),start=start10to13J,numsX=(4,4),numsY=(4,4),gapsX=gapsXmcqJ,gapsY=gapsYmcqJ)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q14to16J,qtags,values = maketemplateMCQ(qNos=list(range(14,17)),start=start14to16J,numsX=(4,4),numsY=(3,3),gapsX=gapsXmcqJ,gapsY=gapsYmcqJ)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q17to20J,qtags,values = maketemplateMCQ(qNos=list(range(17,21)),start=start17to20J,numsX=(4,4),numsY=(4,4),gapsX=gapsXmcqJ,gapsY=gapsYmcqJ)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)


pts={}
pts['J']=roll+q1to4J+q5to9J+q10to13J+q14to16J+q17to20J

start1to4H = (80,574)
start5to8H = (80,799)
start9to12H = (605,183)
start13H = (1105,183)
start14to16H = (605,574)
start17to20H = (605,799)


gapsXintH=(20,38) if kv else (20,37)
gapsXintH20=(20,43)
gapsXmcqH=(20,9)
gapsYmcqH=(18,33)
gapsXintH,gapsXintH20,gapsXmcqH ,gapsYmcqH  = scalePts((gapsXintH,gapsXintH20,gapsXmcqH ,gapsYmcqH ),scalefac)
gapsYintH=(10,31)
gapsYintHRoll=(20,33)
gapsXintHRoll=(39,38)


squad='H'
QTAGS[squad],VALUES[squad] = QTAGS['J'], VALUES['J'] 
QTAGS['H'], VALUES['H']  = QTAGS['J'], VALUES['J'] 

q1to4H,qtags,values = maketemplateMCQ(qNos=list(range(1,5)),start=start1to4H,numsX=(4,4),numsY=(4,4),gapsX=gapsXmcqH,gapsY=gapsYmcqH)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q5to8H,qtags,values = maketemplateMCQ(qNos=list(range(5,9)),start=start5to8H,numsX=(4,4),numsY=(4,4),gapsX=gapsXmcqH,gapsY=gapsYmcqH)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q9to12H,qtags,values  = maketemplateINT(qNos=list(range(9,13)),start=start9to12H,numsX=(2,8),numsY=(1,10),gapsX=gapsXintH20,gapsY=gapsYintH)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)
q13H,qtags,values  = maketemplateINT(qNos=list(range(13,14)),start=start13H,numsX=(2,2),numsY=(1,10),gapsX=gapsXintH20,gapsY=gapsYintH)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q14to16H,qtags,values = maketemplateMCQ(qNos=list(range(14,17)),start=start14to16H,numsX=(4,4),numsY=(3,3),gapsX=gapsXmcqH,gapsY=gapsYmcqH)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

q17to20H,qtags,values = maketemplateMCQ(qNos=list(range(17,21)),start=start17to20H,numsX=(4,4),numsY=(4,4),gapsX=gapsXmcqH,gapsY=gapsYmcqH)
QTAGS[squad],VALUES[squad]   = merge_dicts(QTAGS[squad],qtags) , merge_dicts(VALUES[squad],values)

pts['H']=roll+q1to4H+q5to8H+q9to12H+q13H+q14to16H+q17to20H
