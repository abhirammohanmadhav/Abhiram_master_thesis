import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from shapely.geometry import LineString

def generate_geometry_start(bump_center_1,bump_center_2,bump_radius_1,bump_radius_2,bump_radius_min,bump_radius_max):

    #parameters
    diameter=20
    start_x=0
    end_x=200
    
    # bump_radius_1 = 2
    # bump_radius_2 = 2
    # bump_center_1 = 75
    # bump_center_2 = 125
    # bump_radius_min = 0
    # bump_radius_max = 33
    
    
    bump_radius_1=bump_radius_1
    bump_radius_2=bump_radius_2
    
    #List for bump function
    bump_function_list1=[]
    
    #bump function
    for i in range(-99,100):
    
       bp = np.exp(1/((i/100)**2-1))
       bump_function_list1.append([(i/100),bp])
    
       #strecken und stauchen
       bump_function_array1=np.array(bump_function_list1)
    
       bump_function_array_mod_1=bump_function_array1
       bump_function_array_mod_1[:,0]=bump_function_array_mod_1[:,0]*bump_radius_1+bump_center_1+20 #75+18=93
       # print(bump_function_array_mod_1[:,0]*bump_radius_1+bump_center_1)
       bump_function_array_mod_1[:,1]=bump_function_array_mod_1[:,1]*bump_radius_1
    first_line = LineString(np.column_stack((bump_function_array_mod_1[:,0], bump_function_array_mod_1[:,1])))
    
    
    #List for bump function
    bump_function_list2=[]
    
    #bump function
    for i in range(-99,100):
    
       bp = np.exp(1/((i/100)**2-1))
       bump_function_list2.append([(i/100),bp])
    
    
       #strecken und stauchen
       bump_function_array2=np.array(bump_function_list2)
    
       bump_function_array_mod_2=bump_function_array2
       bump_function_array_mod_2[:,0]=bump_function_array_mod_2[:,0]*bump_radius_2+bump_center_2-20 #125-18=107 when bpr=0
       # print(bump_function_array_mod_2[:,0]*bump_radius_2+bump_center_2)
       bump_function_array_mod_2[:,1]=bump_function_array_mod_2[:,1]*bump_radius_2
    
    Second_line = LineString(np.column_stack((bump_function_array_mod_2[:,0], bump_function_array_mod_2[:,1])))
    intersection = first_line.intersection(Second_line)
    #x, y = intersection.xy
    intersection = np.array(intersection)
    #print(np.shape(intersection), intersection)
    #print(x, y)
    bfa1_x = bump_function_array1[:,0]
    bfa2_x = bump_function_array2[:,0]
    bfa1_y = bump_function_array1[:,1]
    bfa2_y = bump_function_array2[:,1]
    if intersection.shape[0] != 0:
    
        x = intersection[0]
        #print('shape of x coordinate is ', np.shape(x))
        #x1 = np.full((np.size(bfa1_x)), x, dtype=float)
        #x2 = np.full((np.size(bfa2_x)), x, dtype=float)
        y = intersection[1]
        #y1 = np.full((np.size(bfa1_y)), y, dtype=float)
        #y2 = np.full((np.size(bfa2_y)), y, dtype=float)
        #print(x.shape, y)
        if (np.shape(x) > (1,)) and (np.shape(y) > (1,)):
            delta_x1 = np.abs(bfa1_x - x[0]).tolist()
            #print(delta_x1)
            delta_x2 = np.abs(bfa2_x - x[0]).tolist()
        
            delta_y1 = np.abs(bfa1_y - y[1]).tolist()
    
            delta_y2 = np.abs(bfa2_y - y[1]).tolist()
        else:
            delta_x1 = np.abs(bfa1_x - x).tolist()
            #print(delta_x1)
            delta_x2 = np.abs(bfa2_x - x).tolist()
        
            delta_y1 = np.abs(bfa1_y - y).tolist()
    
            delta_y2 = np.abs(bfa2_y - y).tolist()
        
        idx2 = delta_y2.index(min(delta_y2))
        idx1 = bump_function_array_mod_1.shape[0]-delta_y1.index(min(delta_y1))    
        
        bfam1 = bump_function_array_mod_1[:idx1,:]
        bfam2 = bump_function_array_mod_2[idx2:,:]
        #Write files
        #Inlet
        f = open("Inlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(start_x)+" "+"0"+"\n")
        f.write(str(start_x)+" "+str(diameter)+"\n")
        f.close()
        
        #Outlet
        f = open("Outlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(end_x)+" "+str(diameter)+"\n")
        f.write(str(end_x)+" "+"0"+"\n")
        f.close()
        
        #Channel
        f = open("LS_Channel.0", "w")
        f.write(str(bfam1.shape[0]+bfam2.shape[0]+5)+"\n")
        f.write("\n")
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+"0"+"\n")
        f.write(str(bfam1[0,0])+" "+"0"+"\n")
        
        #bump 1 start 
        #f.write(str(bump_function_array_mod_1[bump_function_array_mod_1.shape[0]-1,0])+" "+"0"+"\n")
        
        for k in range(1,bfam1.shape[0]-1):
           # print(k)
           
           f.write(str(bfam1[k,0])+" "+str(bfam1[k,1])+"\n")
           # print(bfam11[0])
        
        f.write(str(bfam1[-1,0])+" "+str(bfam1[-1,1])+"\n")
        #bump 1 end
    
        #bump 2 start
        
        for j in range(0,bfam2.shape[0]-1):
        
           f.write(str(bfam2[j,0])+" "+str(bfam2[j,1])+"\n")
           # print(bfam21[j])
           # print(bfam2.shape[0])
        
        f.write(str(bfam2[-1, 0])+" "+"0"+"\n")
        
        #bump 2 end
    
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.close()
        
        #Normalize bump_center and bump_radius
        bump_radius_1=(bump_radius_1-bump_radius_min)/(bump_radius_max-bump_radius_min)
        bump_radius_2=(bump_radius_2-bump_radius_min)/(bump_radius_max-bump_radius_min)
        
        
    elif intersection.shape[0] == 0:
        #Write files
        #Inlet
        f = open("Inlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(start_x)+" "+"0"+"\n")
        f.write(str(start_x)+" "+str(diameter)+"\n")
        f.close()
         
        #Outlet
        f = open("Outlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(end_x)+" "+str(diameter)+"\n")
        f.write(str(end_x)+" "+"0"+"\n")
        f.close()
         
        #Channel
        f = open("LS_Channel.0", "w")
        f.write(str(bump_function_array_mod_1.shape[0]+bump_function_array_mod_2.shape[0]+4)+"\n")
        f.write("\n")
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+"0"+"\n")
        f.write(str(bump_function_array1[0,0])+" "+"0"+"\n")
        #bump 1 start      
        for k in range(1,bump_function_array1.shape[0]-1):
           f.write(str(bump_function_array_mod_1[k,0])+" "+str(bump_function_array_mod_1[k,1])+"\n")
    
        # #bump 1 end
     
        # #bump 2 start
    
        for j in range(0,bump_function_array2.shape[0]-1):
           f.write(str(bump_function_array_mod_2[j,0])+" "+str(bump_function_array_mod_2[j,1])+"\n")
         
        f.write(str(bump_function_array_mod_2[bump_function_array_mod_2.shape[0]-1,0])+" "+"0"+"\n")
        #bump 2 end
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.close()
        
        #Normalize bump_center and bump_radius
        bump_radius_1=(bump_radius_1-bump_radius_min)/(bump_radius_max-bump_radius_min)
        bump_radius_2=(bump_radius_2-bump_radius_min)/(bump_radius_max-bump_radius_min)
    return bump_radius_1, bump_radius_2
# bpr1, bpr2 = generate_geometry_start(75,125,18,10,0,33)

def generate_geometry_train(nr_iter,nr_steps,bump_center_1,bump_center_2,previous_radius_1,previous_radius_2,radius_change_1,radius_change_2,bump_radius_min,bump_radius_max):

    #parameters
    diameter=20
    start_x=0
    
    end_x=200
    
    #Denormalize bump_radius
    previous_radius_1=previous_radius_1*(bump_radius_max-bump_radius_min)+bump_radius_min
    previous_radius_2=previous_radius_2*(bump_radius_max-bump_radius_min)+bump_radius_min
    
    #Change radius
    bump_radius_1=previous_radius_1+radius_change_1
    bump_radius_2=previous_radius_2+radius_change_2

    death_min1 = False
    death_max1 = False
    death_min2 = False
    death_max2 = False
    if bump_radius_1 >= bump_radius_max: 
       death_max1=True

    elif bump_radius_2 >= bump_radius_max:
       death_max2 = True
    
    if bump_radius_1 <= bump_radius_min:
       death_min1 = True

    elif bump_radius_2 <= bump_radius_min:
       death_min2 = True

    bump_function_list1=[]
    
    #bump function
    for i in range(-99,100):
    
       bp = np.exp(1/((i/100)**2-1))
       bump_function_list1.append([(i/100),bp])
    
       #strecken und stauchen
       bump_function_array1=np.array(bump_function_list1)
    
       bump_function_array_mod_1=bump_function_array1
       bump_function_array_mod_1[:,0]=bump_function_array_mod_1[:,0]*bump_radius_1+bump_center_1+20 #75+18=93
       # print(bump_function_array_mod_1[:,0]*bump_radius_1+bump_center_1)
       bump_function_array_mod_1[:,1]=bump_function_array_mod_1[:,1]*bump_radius_1
    first_line = LineString(np.column_stack((bump_function_array_mod_1[:,0], bump_function_array_mod_1[:,1])))
    
    
    #List for bump function
    bump_function_list2=[]
    
    #bump function
    for i in range(-99,100):
    
       bp = np.exp(1/((i/100)**2-1))
       bump_function_list2.append([(i/100),bp])
    
    
       #strecken und stauchen
       bump_function_array2=np.array(bump_function_list2)
    
       bump_function_array_mod_2=bump_function_array2
       bump_function_array_mod_2[:,0]=bump_function_array_mod_2[:,0]*bump_radius_2+bump_center_2-20 #125-18=107 when bpr=0
       # print(bump_function_array_mod_2[:,0]*bump_radius_2+bump_center_2)
       bump_function_array_mod_2[:,1]=bump_function_array_mod_2[:,1]*bump_radius_2
    
    Second_line = LineString(np.column_stack((bump_function_array_mod_2[:,0], bump_function_array_mod_2[:,1])))
    intersection = first_line.intersection(Second_line)
    #x, y = intersection.xy
    intersection = np.array(intersection)
    #print(np.shape(intersection), intersection)
    #print(x, y)
    bfa1_x = bump_function_array1[:,0]
    bfa2_x = bump_function_array2[:,0]
    bfa1_y = bump_function_array1[:,1]
    bfa2_y = bump_function_array2[:,1]
    if intersection.shape[0] != 0:
    
        x = intersection[0]
        #print('shape of x coordinate is ', np.shape(x))
        #x1 = np.full((np.size(bfa1_x)), x, dtype=float)
        #x2 = np.full((np.size(bfa2_x)), x, dtype=float)
        y = intersection[1]
        #y1 = np.full((np.size(bfa1_y)), y, dtype=float)
        #y2 = np.full((np.size(bfa2_y)), y, dtype=float)
        #print(x.shape, y)
        if (np.shape(x) > (1,)) and (np.shape(y) > (1,)):
            delta_x1 = np.abs(bfa1_x - x[0]).tolist()
            #print(delta_x1)
            delta_x2 = np.abs(bfa2_x - x[0]).tolist()
        
            delta_y1 = np.abs(bfa1_y - y[1]).tolist()
    
            delta_y2 = np.abs(bfa2_y - y[1]).tolist()
        else:
            delta_x1 = np.abs(bfa1_x - x).tolist()
            #print(delta_x1)
            delta_x2 = np.abs(bfa2_x - x).tolist()
        
            delta_y1 = np.abs(bfa1_y - y).tolist()
    
            delta_y2 = np.abs(bfa2_y - y).tolist()
        
        idx2 = delta_y2.index(min(delta_y2))
        idx1 = bump_function_array_mod_1.shape[0]-delta_y1.index(min(delta_y1))    
        
        bfam1 = bump_function_array_mod_1[:idx1,:]
        bfam2 = bump_function_array_mod_2[idx2:,:]
        #Write files
        #Inlet
        f = open("Inlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(start_x)+" "+"0"+"\n")
        f.write(str(start_x)+" "+str(diameter)+"\n")
        f.close()
        
        #Outlet
        f = open("Outlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(end_x)+" "+str(diameter)+"\n")
        f.write(str(end_x)+" "+"0"+"\n")
        f.close()
        
        #Channel
        f = open("LS_Channel.0", "w")
        f.write(str(bfam1.shape[0]+bfam2.shape[0]+5)+"\n")
        f.write("\n")
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+"0"+"\n")
        f.write(str(bfam1[0,0])+" "+"0"+"\n")
        
        #bump 1 start 
        #f.write(str(bump_function_array_mod_1[bump_function_array_mod_1.shape[0]-1,0])+" "+"0"+"\n")
        
        for k in range(1,bfam1.shape[0]-1):
           # print(k)
           
           f.write(str(bfam1[k,0])+" "+str(bfam1[k,1])+"\n")
           # print(bfam11[0])
        
        f.write(str(bfam1[-1,0])+" "+str(bfam1[-1,1])+"\n")
        #bump 1 end
    
        #bump 2 start
        
        for j in range(0,bfam2.shape[0]-1):
        
           f.write(str(bfam2[j,0])+" "+str(bfam2[j,1])+"\n")
           # print(bfam21[j])
           # print(bfam2.shape[0])
        
        f.write(str(bfam2[-1, 0])+" "+"0"+"\n")
        
        #bump 2 end
    
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.close()
        
        #Normalize bump_center and bump_radius
        bump_radius_1=(bump_radius_1-bump_radius_min)/(bump_radius_max-bump_radius_min)
        bump_radius_2=(bump_radius_2-bump_radius_min)/(bump_radius_max-bump_radius_min)
        
        
    elif intersection.shape[0] == 0:
        #Write files
        #Inlet
        f = open("Inlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(start_x)+" "+"0"+"\n")
        f.write(str(start_x)+" "+str(diameter)+"\n")
        f.close()
         
        #Outlet
        f = open("Outlet", "w")
        f.write("2"+"\n")
        f.write("\n")
        f.write(str(end_x)+" "+str(diameter)+"\n")
        f.write(str(end_x)+" "+"0"+"\n")
        f.close()
         
        #Channel
        f = open("LS_Channel.0", "w")
        f.write(str(bump_function_array_mod_1.shape[0]+bump_function_array_mod_2.shape[0]+4)+"\n")
        f.write("\n")
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+str(diameter)+"\n")
        f.write("-"+str(diameter)+" "+"0"+"\n")
        f.write(str(bump_function_array1[0,0])+" "+"0"+"\n")
        #bump 1 start      
        for k in range(1,bump_function_array1.shape[0]-1):
           f.write(str(bump_function_array_mod_1[k,0])+" "+str(bump_function_array_mod_1[k,1])+"\n")
    
        # #bump 1 end
     
        # #bump 2 start
    
        for j in range(0,bump_function_array2.shape[0]-1):
           f.write(str(bump_function_array_mod_2[j,0])+" "+str(bump_function_array_mod_2[j,1])+"\n")
         
        f.write(str(bump_function_array_mod_2[bump_function_array_mod_2.shape[0]-1,0])+" "+"0"+"\n")
        #bump 2 end
        f.write(str(end_x+diameter)+" "+"0"+"\n")
        f.close()
        
        #Normalize bump_center and bump_radius
        bump_radius_1=(bump_radius_1-bump_radius_min)/(bump_radius_max-bump_radius_min)
        bump_radius_2=(bump_radius_2-bump_radius_min)/(bump_radius_max-bump_radius_min)
    return bump_radius_1, bump_radius_2, death_max1, death_min1, death_max2, death_min2
