"""

Designed and Developed by-
Udayraj Deshmukh 
https://github.com/Udayraj123

"""

QVals= sorted(QVals)
l=len(QVals)-1
# Find the first 'big' jump and set it as threshold:
# Nope, won't make it automatic - Becomes much easier and safer if guided i.e. given base threshold for black
# --> Find top two jumps at 3 point gap, choose the one with lower value of middle point.
# There's still a change of 'too many responses' situation
# Make use of the fact that the delta between values at detected jumps would be atleast 20
max1,thr1=0,255
for i in range(1,l):
    jump = QVals[i+1] - QVals[i-1]
    if(jump > max1):
        max1=jump
        thr1=QVals[i-1] + jump/2

delta=20
max2,thr2=0,255
for i in range(1,l):
    jump = QVals[i+1] - QVals[i-1]
    d2 = QVals[i-1] + jump/2                
    if(jump > max2 and delta < abs(thr1-d2)):
        max2=jump
        thr2=d2

thresholdRead = min(thr1,thr2)
# blackTHRs.pb
f, ax = plt.subplots() 
thrline=ax.axhline(thresholdRead,color='red',ls='--')
thrline.set_label("newTHR")
ax.bar(range(len(QVals)),QVals);
plt.show()