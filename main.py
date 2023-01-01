# %%
print('Link Hysteresis Energy optimization')

# %% importing libraries
import os
import sys
import comtypes.client
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pygad
#import cProfile # debugging 


# %%------------------------------------ INPUT--------------------------------------

# initial solution

Y_strength = [500, 500, 500, 500, 500]

Cycle_limit = 80 # Limit on hysteresis loops number

Link_names = ['LINK1', 'LINK2','LINK3','LINK4','LINK5'] # Links names

num_links = 40 # set to 20 for 5 story # Number of links
Link_labels = range(1, num_links+1)
Link_labels = [format(x, 'd') for x in Link_labels] # generates list with Link numbers

Story_height = 3.0 # story heigh in meters
drift_limit = 5 # interstory drift in %

Ductility_limit = 8 # Ductility limit

# joints number for story , Set to ["2", "16", "27", "38", "49"] for 5 story

joints = ["2", "16", "27", "38", "49", "60", "71", "82", "93", "104"] 

Load_case_name = "ARTIF1" # Nonlinear load case name


# %% SAP OAPI initiation
SapObject = comtypes.client.GetActiveObject("CSI.SAP2000.API.SapObject")
SapModel = SapObject.SapModel
SapModel.SetModelIsLocked(False)
# %% setting link parameters 
def link_option(Y_strength, L_name):
    """
    Modifies the yield strength for M3 of the selected link element
    Arguments:
        Y_strength: new yield strength
        L_name: Link name
    """
    DOF = [False, False, False, False, False, False]
    Fixed = [False, False, False, False, False, False]
    NonLinear = [False, False, False, False, False, False]
    Ke = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Ce = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    K =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Yield = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Ratio = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Exp =[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    DOF[0] = True
    Fixed[0] = True

    DOF[1] = True
    Fixed[1] = True

    DOF[2] = True
    Fixed[2] = True

    DOF[3] = True
    Fixed[3] = True

    DOF[3] = True
    Fixed[3] = True

    DOF[4] = True
    Fixed[4] = True

    DOF[5] = True
    NonLinear[5] = True
    Ke[5] = 100000
    Ce[5] = 0
    K[5] = 100000
    Yield[5] = Y_strength
    Ratio[5] = 0.03
    Exp[5] = 2
    ret = SapModel.PropLink.SetPlasticWen(L_name, DOF, Fixed, NonLinear, Ke, Ce, K, Yield, Ratio, Exp, 2, 0)
    ret = SapModel.PropLink.SetPDelta(L_name, [0.5, 0.5, 0.5, 0.5])
# %% hysteresis loop number

def cycle_number(result):
    """
    Calculates the number of load-unload cycles
    Arguments :
        result : pandas DataFrame with Force-Displacment results
    Returns :
        c : number of cycles
        result : modifies results DataFrame
    """
    result['q2'] = 0
    result['c'] = 0
    q1 = 0
    q2 = 0
    c = 0
    for i in range(len(result)-1):
        if result['Force'][i] >= 0.0 and result['Desplacment'][i] >= 0.0 : 
            q2 = 1
            result['q2'][i] = q2
        if result['Force'][i] < 0.0 and result['Desplacment'][i] > 0.0 : 
            q2 = 2
            result['q2'][i] = q2
        if result['Force'][i] < 0.0 and result['Desplacment'][i] < 0.0 : 
            q2 = 3
            result['q2'][i] = q2
        if result['Force'][i] > 0.0 and result['Desplacment'][i] < 0.0 : 
            q2 = 4
            result['q2'][i] = q2
        
        
        if q1 > q2 and q2 ==1:
            c += 1
        result['c'][i] = c

        q1 = q2
    return c, result



# %% Area under function curve calculation
def Area(result):
    """
    Calculates the area under the function curve for a set of points using Trapezoidal rule
    Arguments : 
        result : DataFrame with the function data points
    Returns : 
        Energy : The area under the function curve
    """
    #Energy = np.abs(result['Desplacment'][len(result)-1]*result['Force'][0] - result['Desplacment'][0]*result['Force'][len(result)-1])/2
    Energy = 0
    for i in range(len(result)-1):
        Area = result['Desplacment'][i]*result['Force'][i+1] - result['Desplacment'][i+1]*result['Force'][i]
        #Area = (result['Force'][i] + result['Force'][i+1]) / 2 * (result['Desplacment'][i+1]- result['Desplacment'][i])
        Energy = Energy +  Area
    return Energy

# %% Hysteresis energy for a selected link

def energy_func(result):
    c, result = cycle_number(result)
    E = Area(result)
    return 0.5*np.abs(E), c

# %% Get analysis results

def get_data(L_num, save_path,Plot_graph = True):
    """
    Gets the analysis results
    Arguments:
        L_num: Link number
        save_path: Path to save hysteresis plots of the selected link
        Plot_graph : If true saves the hysteresis plots and results of the selected link
    Returns:
        result : Pandas DataFrame with the analysis results
        Energy : Hysteresis energy of the selected link
        c : number of hysteresis loops
    """
    eItemTypeElm = 1
    NumberResults = 0
    Obj =[]
    Elm =[]
    PointElm =[]
    LoadCase =[]
    StepType =[]
    StepNum =[]
    P =[]
    V2 =[]
    V3 =[]
    T =[]
    M2 =[]
    M3 =[]

    U1=[]
    U2=[]
    U3=[]
    R1=[]
    R2=[]
    R3=[]
    SapModel.Results.Setup.SetOptionModalHist(2)
    SapModel.Results.Setup.SetOptionDirectHist(2)
    [NumberResults, Obj, Elm, LoadCase,PointElm, StepType, StepNum, P, V2, V3, T, M2, M3, ret] = SapModel.Results.LinkForce(L_num, eItemTypeElm, NumberResults, Obj, Elm, PointElm, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret]= SapModel.Results.LinkDeformation(L_num, eItemTypeElm, NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
    result = pd.DataFrame()

    result['Desplacment'] = R3
    result['Force'] = M3[0:len(M3):2]

    if Plot_graph:
        plt.plot(result['Desplacment'], result['Force'])
        plt.grid()
        save_path = os.path.join(save_path, 'plots') 
        os.makedirs(save_path, exist_ok=True) 
        save_path_csv = save_path + '/'+ str(L_num) + ".csv"
        save_path = save_path + '/'+ str(L_num) + ".png"
        result.to_csv(save_path_csv, index = False)
        plt.savefig(save_path)
        plt.close()
 
        
    u = ductility(result)
    Energy, c = energy_func(result)


    return result, Energy, c, u


# %% inter story drift check
def drift_check(story_ids, Story_height, drift_limit, Y_strength):
    """ 
    Checks the inter story drift limitation 
    Arguments : 
        story_ids : Joints ids from each story
        Story_height : story height in meters
        drift_limit : drift limitation in %
    Returns : 
        Check : Boolean, True if the limitation is exceeded
    """
    Check = False
    U = []
    for joint in story_ids :
        eItemTypeElm = 1
        NumberResults = 0
        Obj =[]
        Elm =[]
        PointElm =[]
        LoadCase =[]
        StepType =[]
        StepNum =[]


        U1=[]
        U2=[]
        U3=[]
        R1=[]
        R2=[]
        R3=[]
        SapModel.Results.Setup.SetOptionModalHist(1)
        SapModel.Results.Setup.SetOptionDirectHist(1)
        [NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret]= SapModel.Results.JointDispl(joint, eItemTypeElm, NumberResults, Obj, Elm, LoadCase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
        U.append(np.abs(U1[0]))
    U = np.array(U)
    drift = []
    drift.append(np.abs(U[0])/Story_height*100)
    for i in range(1,len(U)):
        inter_story_drift = np.abs(U[i]-U[i-1])/Story_height*100
        drift.append(inter_story_drift)
    drift_pd = pd.DataFrame(list(zip(range(1, len(drift)+1), drift)),  columns= ['Story', 'Drift'])
    print(drift_pd)
    save_path_csv = os.path.join("results/", str(Y_strength))  + "/Drift.csv"
    drift_pd.to_csv(save_path_csv, index = False)
    # print("Max inter story drift : ", np.max(drift), "%")
    if np.max(drift) > drift_limit : 
        Check = True
    return Check

# %% Ductility 

def ductility(result):

    a1 = result.Force[1]/result.Desplacment[1]
    b1 = 1 
    c1 = 0

    a2 = (result.Force[result.Force.idxmax()]-result.Force[result.Force.idxmax()-1])/(result.Desplacment[result.Force.idxmax()]-result.Desplacment[result.Force.idxmax()-1])
    b2 = 1
    c2 = result.Force[result.Force.idxmax()] - a2 * result.Desplacment[result.Force.idxmax()]


    Xx = (b1*c2-b2*c1)/(a1*b2-a2*b1)
    Yy = -1*(a2*c1-a1*c2)/(a1*b2-a2*b1)


    u = result.Desplacment[result.Force.idxmax()]/Xx

    if u <1 : 
        u = 1

    return u

# %% Main function



def main(Y_strength, Link_names, Link_labels, Load_case_name, save_data = True):
    """
    Calculates the total hysteresis energy of the structure for a selected yield strength values
    Aeguments : 
        Y_strength : List of Yield strengths for each link
        Link_names : Link section names
        Link_labels : Links labels
        Load_case_name : Non-linear load case name
        Cycle_limit : Limit for load-unload cycles
        save_data : If True, saves the hysteresis plots for each link element in /plots
    Returns :
        Total hysteresis energy of the structure
    """
    print(Y_strength)
    SapModel.SetModelIsLocked(False)
    Link_list = Link_names
    Link_numbers = Link_labels

    for link_name in Link_list:
        link_option( Y_strength[Link_list.index(link_name)], link_name)

    # Run the analysis
    SapModel.Analyze.RunAnalysis()
    SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    # Select load case
    SapModel.Results.Setup.SetCaseSelectedForOutput(Load_case_name)
    # Select how to get the results (2 for Step by Step)
    SapModel.Results.Setup.SetOptionModalHist(2)
    SapModel.Results.Setup.SetOptionDirectHist(2)


    E = []
    U = []
    C = []
    plot_dirc = os.path.join("results/", str(Y_strength)) 
    os.makedirs(plot_dirc, exist_ok=True) 

    for num in Link_numbers:
        r, e, c, u = get_data(num, plot_dirc, Plot_graph= save_data)
        if u!=1:
            U.append(u)
        E.append(e)
        C.append(c)
    C_max = []
    for i in range(0, len(C)):
        if i % 4 ==0:
            C_max.append(np.max(C[i:i+4]))
    
    C_max = pd.DataFrame(list(zip(range(1, len(C_max)+1), C_max)), columns= ['Story', 'Max Fatigue cycles'])
    print(C_max)
    save_path_csv = plot_dirc  + "/Max_Fatigue_cycles.csv"
    C_max.to_csv(save_path_csv, index = False)

    U_max = []
    for i in range(0, len(U)):
        if i % 4 ==0:
            U_max.append(np.max(U[i:i+4]))
    
    U_max = pd.DataFrame(list(zip(range(1, len(U_max)+1), U_max)), columns= ['Story', 'Max Ductility'])
    print(U_max)
    save_path_csv = plot_dirc  + "/Max_Ductility.csv"
    U_max.to_csv(save_path_csv, index = False)


    #print('Fatigue cycles  : ','Max =',np.max(C),'Mean = ', np.mean(C))
    #print('Ductility : ','Max =',np.max(U),'Mean = ', np.mean(U))
    
    drift_Check = drift_check(joints, Story_height, drift_limit, Y_strength)
    E_tot = np.sum(E)
    
    print('E = ',E_tot)
    return E_tot, C, U, drift_Check





# %% The benchmark

print('---------------------Our initial model :-----------------')
benchmark = main(Y_strength, Link_names, Link_labels, Load_case_name)

# %% Genetic Algorithm
def fitness_func(solution, solution_idx):
    """
    Only allows to solutions better than our initial model (benchmark) to continue to the next generation
    """
    fitness_score, C, U,  drift_Check= main(solution, Link_names, Link_labels, Load_case_name)
    
    if fitness_score < benchmark[0] :
        fitness_score = 0.0

    if drift_Check:
        fitness_score = 0.0
    
    if np.max(C) > Cycle_limit :
        fitness_score = 0.0

    if np.max(U) > Ductility_limit : 
        fitness_score = 0.0
    
    print('fitness score = ',fitness_score)
    
    return fitness_score

last_fitness = 0

def on_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness))
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

fitness_function = fitness_func

num_generations = 7 #iteratinos
num_parents_mating = 2 

sol_per_pop = 10 # solutions per iteration
num_genes = len(Y_strength) 

init_range_low = 80 # lowest solution limit
init_range_high = 600 # highest solution limit

parent_selection_type = "sss" # rank_selection()"sss"
keep_parents = -1 # keep all parents
                    

crossover_type = "single_point" #"two_points" #"uniform" #"single_point"

mutation_type =  "random" #"swap"
mutation_percent_genes = 60

ga_instance = pygad.GA(on_generation=on_generation,
gene_type=int,
                        num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       save_solutions=True)

# %% Start the optimisation
#input('Press enter to start the optimisation...')
print('---------------The optimisation starts :------------------')
ga_instance.run()

# %% Solutions

plot_dirc = os.path.join("results/", "Final_results") 
os.makedirs(plot_dirc, exist_ok=True) 

Solutions_fitness = ga_instance.solutions_fitness
Solutions_fitness = pd.DataFrame(list(zip(range(1, len(Solutions_fitness)+1), Solutions_fitness)), columns= ['Number', 'Fitness'])
Solutions_fitness.loc[-1]=[0, benchmark[0]]
Solutions_fitness.index = Solutions_fitness.index + 1
Solutions_fitness = Solutions_fitness.sort_index()
Solutions_fitness = Solutions_fitness[Solutions_fitness.Fitness !=0]
Solutions_fitness.reset_index(drop=True, inplace=True)

save_path_csv = plot_dirc  + "/Solutions_Fitness.csv"
Solutions_fitness.to_csv(save_path_csv, index = False)

save_path = plot_dirc + '/Solutions_Fitness' + ".png"

first_point = Solutions_fitness.iloc[0].Fitness
plt.plot(Solutions_fitness.Number, Solutions_fitness.Fitness)
plt.plot(0, first_point, 'o', markersize = 10 )
plt.text(0, first_point, 'Initial model', horizontalalignment='center',
     verticalalignment='center')
plt.grid()
plt.xlabel("Solutions")
plt.ylabel("Fitness score")
plt.savefig(save_path)
plt.close()

# %% best solutions 

Best_fitness = ga_instance.best_solutions_fitness
Best_fitness = pd.DataFrame(list(zip(range(1, len(Best_fitness)+1), Best_fitness)), columns= ['Generation', 'Fitness'])
Best_fitness.loc[-1]=[0, benchmark[0]]
Best_fitness.index = Best_fitness.index + 1
Best_fitness = Best_fitness.sort_index()
Best_fitness = Best_fitness[Best_fitness.Fitness !=0]
Best_fitness.reset_index(drop=True, inplace=True)

save_path_csv = plot_dirc  + "/Generations_Fitness.csv"
Best_fitness.to_csv(save_path_csv, index = False)

save_path = plot_dirc + '/Generations_Fitness' + ".png"

first_point = Best_fitness.iloc[0].Fitness
plt.plot(Best_fitness.Generation, Best_fitness.Fitness)
plt.plot(0, first_point, 'o', markersize = 10 )
plt.text(0, first_point, 'Initial model', horizontalalignment='center',
     verticalalignment='center')
plt.grid()
plt.xlabel("Generations")
plt.ylabel("Fitness score")
plt.savefig(save_path)
plt.close()


# %% Solution
input("Press enter to pass to the best solution ...")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# %%
