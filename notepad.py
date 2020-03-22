import sys,getopt

opts,args=getopt.getopt(sys.argv[1:],'n:o:')
for opt,arg in opts:
    if opt=='-n':
        one=arg
    if opt=='-o':
        two=arg
print("\n")
print(one,two)