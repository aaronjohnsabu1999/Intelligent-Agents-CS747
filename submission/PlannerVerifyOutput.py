#! /usr/bin/python
import random,argparse,sys,subprocess,os,time
parser = argparse.ArgumentParser()
import numpy as np
random.seed(0)

input_file_ls = ['data/mdp/continuing-mdp-10-5.txt','data/mdp/continuing-mdp-2-2.txt','data/mdp/continuing-mdp-50-20.txt','data/mdp/episodic-mdp-10-5.txt','data/mdp/episodic-mdp-2-2.txt','data/mdp/episodic-mdp-50-20.txt']

class VerifyOutputPlanner:
    def __init__(self,algorithm,print_error):
        algorithm_ls = list()
        if algorithm=='all':
            algorithm_ls+=['hpi','vi','lp']
        else:
            algorithm_ls.append(algorithm)
            
        for algo in algorithm_ls:
            print('verify output',algo)
            counter = 1    
            
            for in_file in input_file_ls:
                dt = time.time()
                print("\n\n","-"*100)
                cmd_planner = "python","planner.py","--mdp",in_file,"--algorithm",algo
                print('test case',str(counter)+algo,":\t"," ".join(cmd_planner))
                counter+=1
                cmd_output = subprocess.check_output(cmd_planner,universal_newlines=True)
                self.verifyOutput(cmd_output,in_file,print_error)
                print('Time taken for iteration = ', str(time.time()-dt))
                
    def verifyOutput(self,cmd_output,in_file,pe):
        sol_file = in_file.replace("continuing","sol-continuing").replace("episodic","sol-episodic")
        base = np.loadtxt(sol_file,delimiter=" ",dtype=float)
        output = cmd_output.split("\n")
        nstates = base.shape[0]
        est = [i.split() for i in output if i!='']
        
        mistakeFlag = False
        #Check1: Checking the number of lines printed
        if not len(est)==nstates:
            mistakeFlag = True
            print("\n","*"*10,"Mistake:Exact number of line in standard output should be",nstates,"but have",len(est),"*"*10)
            
        #Check2: Each line shoul have only two values
        for i in range(len(est)):
            if not len(est[i])==2:
                mistakeFlag = True
                print("\n","*"*10,"Mistake: On each line you should print only value,policy for a state","*"*10)
                break
        
        if not mistakeFlag:
            print("ALL CHECKS PASSED!")
        else:
            print("You haven't printed output in correct format.")
        
        pe_ls = ['no','NO','No','nO']
        if pe not in pe_ls:
            if not mistakeFlag:
                print("Calculating error of your value function...")
            else:
                print("\nExiting without calculating error of your value function")
                return
            #calculating the error
            for i in range(len(est)):
                est_V = float(est[i][0]);base_V = float(base[i][0])
                print("%10.6f"%est_V,"%10.6f"%base_V,"%10.6f"%abs(est_V-base_V),end="\t")
                if abs(est_V-base_V) <= (10**-4):
                    print("OK")
                else:
                    print("\tNot OK")


if __name__ == "__main__":
    parser.add_argument("--algorithm",type=str,default="all")
    parser.add_argument("--pe",type=str,default="yes")
    args = parser.parse_args()
    start = time.time()
    algo = VerifyOutputPlanner(args.algorithm,args.pe)
    print('Total time taken = ', str(time.time()-start))