import os,random,subprocess
random.seed(2)
fname_ls = ["outputDataT1.txt","outputDataT2.txt"]
number_lines_dict = {"outputDataT1.txt":3600,"outputDataT2.txt":1800}
algo_dict = {"outputDataT1.txt":['epsilon-greedy','ucb','kl-ucb','thompson-sampling'],"outputDataT2.txt":['thompson-sampling','thompson-sampling-with-hint']}
for fname in fname_ls:
    errorFlag = False
    
    #Check if file exists
    print('\n\n-------- verifying', fname ,'data ---------------') 
    try:
        f = open(fname, "r")
        line_ls = [line for line in f.read().split("\n")]
        len_line_ls = len(line_ls)
        #Check for number of lines
        if not len_line_ls==number_lines_dict[fname]:
            print("\n","*"*10,"Mistake:number of lines in the output data file should be",number_lines_dict[fname],"but have ",len_line_ls,"*"*10,"\n")
            errorFlag = True
        
        
        lists = [];set_main=set()
        for i in range(4):
            lists.append([])

        for line in line_ls:
            line=line.replace("\n","").split(", ")
            if not len(line)==6:
                print("\n","*"*10,"Mistake: Wrong line printed",line,"*"*10,"\n")
                continue
            lists[0].append(line[0])  #instance
            lists[1].append(line[1])  #algo
            lists[2].append(int(line[2])) #randomSeed
            lists[3].append(int(line[4])) # horizon
            
            set_main.add(line[0]+"--"+line[1]+"--"+line[2]+"--"+line[4])
        
        if not len(set_main)==number_lines_dict[fname]:
            print("\n","*"*10,"Mistake: You didn't print all the combinations. Need ",number_lines_dict[fname],"but printed ",len(set_main),"*"*10,"\n")
            errorFlag = True
            #for item in set_main:
            #    print(item)
            
        for algo in algo_dict[fname]:
            if not lists[1].count(algo)==900:
                print("\n","*"*10,"Mistake: Each algorithm should be run 900 times i.e 3 instances times 50 seeds times 6 horizons but ",algo, "has run only",lists[1].count(algo)," times","*"*10,"\n")
        
          
        
        #insti = list(set(lists[0]));algo = list(set(lists[1]));hori = list(set(lists[3]));rs = list(set(lists[2]))
        #rs.sort();insti.sort();algo.sort()
            
        
        #try to reproduce random 10 data points
        for i in range(10):
            line_str = line_ls[random.randint(0,len_line_ls)]
            line = line_str.replace("\n","").split(",")
            orig_REG = line[-1].strip()
            #print(line_str) #['../instances/i-3.txt', ' thompson-sampling', ' 4', ' 0.02', ' 6400', ' 58'
           
            cmd = "python","bandit.py","--instance",line[0].strip(),"--algorithm",line[1].strip(),"--randomSeed",line[2].strip(),"--epsilon",line[3].strip(),"--horizon",line[4].strip()
            print("running",cmd)
            reproduced_str = subprocess.check_output(cmd,universal_newlines=True)
            reproduced = reproduced_str.replace("\n","").split(",")
            rep_REG =  reproduced[-1].strip()
            
            if not rep_REG==orig_REG:
                print("\n","*"*10,"Mistake: Unable to reproduce result for ",line_str," orignal="+orig_REG+" reproduced="+rep_REG,"\t","*"*10,"\n")
                #print(line_str)
                errorFlag = True
        f.close()
    except:
        print("*"*10,"Mistake:There is no file named", fname,"*"*10)
        errorFlag = True
    
    #print("\n\n**********")
    #line = ['../instances/i-3.txt', 'thompson-sampling', ' 4', ' 0.02', ' 6400', ' 58']
    #cmd = "python","bandit.py","--instance",line[0],"--algorithm",line[1],"--randomSeed",line[2].strip(),"--epsilon",line[3].strip(),"--horizon",line[4].strip()
    #subprocess.run(cmd)
if errorFlag:
    print("\n","*"*10,"Some issue with your submission data","*"*10,"\n")
else:
    print("Everything is Okay")