import numpy

k=3
n=6
m=8
p=m*n

def recConvol(k,n,m,p):
    h=numpy.sqrt((m*p)/n)
    v=(p/h)
    Sh=(m/h)
    Sv=(n/v)
    padH=((k-Sh)/2)
    padV=((k-Sv)/2)


