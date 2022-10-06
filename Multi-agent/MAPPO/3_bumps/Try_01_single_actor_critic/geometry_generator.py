import numpy as np
import matplotlib.pyplot as plt

def generate_geometry_start(bump_center_1,bump_center_2,bump_center_3,bump_radius_1,bump_radius_2,bump_radius_3,bump_radius_min,bump_radius_max):
    
    #parameters
    diameter=25
    start_x=0
    end_x=250
    # bump_center_1 = 75
    # bump_center_2 = 125
    # bump_center_3 = 175
    # bump_radius_1 = 33
    # bump_radius_2 = 33
    # bump_radius_3 = 33
    # bump_radius_min = 0
    # bump_radius_max = 33
    
    # bump_radius_1=bump_radius_1
    # bump_radius_2=bump_radius_2
    # bump_radius_3=bump_radius_3
    
    #List for bump function
    bump_function_list=[]
    
    #bump function
    for i in range(-99,100):
    
       bp = np.exp(1/((i/100)**2-1))
       bump_function_list.append([(i/100),bp])
    
       #strecken und stauchen
       bump_function_array=np.array(bump_function_list)
    
       bump_function_array_mod_1=bump_function_array
       bump_function_array_mod_1[:,0]=bump_function_array_mod_1[:,0]*bump_radius_1+bump_center_1
       bump_function_array_mod_1[:,1]=(bump_function_array_mod_1[:,1]*-bump_radius_1)+diameter
    
    #List for bump function
    bump_function_list=[]
    
    #bump function
    for i in range(-99,100):
    
       bp = np.exp(1/((i/100)**2-1))
       bump_function_list.append([(i/100),bp])
    
       #strecken und stauchen
       bump_function_array=np.array(bump_function_list)
    
       bump_function_array_mod_2=bump_function_array
       bump_function_array_mod_2[:,0]=bump_function_array_mod_2[:,0]*bump_radius_2+bump_center_2
       bump_function_array_mod_2[:,1]=bump_function_array_mod_2[:,1]*bump_radius_2
    
    #List for bump function
    bump_function_list=[]
    
    #bump function
    for i in range(-99,100):
    
       bp = np.exp(1/((i/100)**2-1))
       bump_function_list.append([(i/100),bp])
    
       #strecken und stauchen
       bump_function_array=np.array(bump_function_list)
    
       bump_function_array_mod_3=bump_function_array
       bump_function_array_mod_3[:,0]=bump_function_array_mod_3[:,0]*bump_radius_3+bump_center_3
       bump_function_array_mod_3[:,1]=(bump_function_array_mod_3[:,1]*-bump_radius_3)+diameter
    
    fig, ax = plt.subplots()
    ax.plot(bump_function_array_mod_1[:,0], bump_function_array_mod_1[:,1])
    ax.plot(bump_function_array_mod_2[:,0], bump_function_array_mod_2[:,1])
    ax.plot(bump_function_array_mod_3[:,0], bump_function_array_mod_3[:,1])
    
    ax.set_xlim(0, end_x)
    ax.set_ylim(0, diameter)
    fig.set_size_inches(20, 3)
    plt.close()
    
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
    f.write(str(3*bump_function_array.shape[0]+3)+"\n")
    f.write("\n")
    f.write(str(end_x+diameter)+" "+"0"+"\n")
    f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
    #bump 1 start 
    #f.write(str(bump_function_array_mod_1[bump_function_array_mod_1.shape[0]-1,0])+" "+"0"+"\n")
    
    for l in range(bump_function_array.shape[0]-1,1,-1):
        f.write(str(bump_function_array_mod_3[l,0])+" "+str(bump_function_array_mod_3[l,1])+"\n")
    f.write(str(bump_function_array_mod_3[0,0])+" "+str(diameter)+"\n")
    # f.write(str(bump_function_array_mod_1[-1,0])+" "+str(diameter)+"\n")
    
    for k in range(bump_function_array.shape[0]-1,1,-1):
       f.write(str(bump_function_array_mod_1[k,0])+" "+str(bump_function_array_mod_1[k,1])+"\n")
    
    f.write(str(bump_function_array_mod_1[0,0])+" "+str(diameter)+"\n")
    #bump 1 end
    f.write("-"+str(diameter)+" "+str(diameter)+"\n")
    f.write("-"+str(diameter)+" "+"0"+"\n")
    #bump 2 start
    f.write(str(bump_function_array_mod_2[0,0])+" "+"0"+"\n")
    
    for j in range(1,bump_function_array.shape[0]-1):
       f.write(str(bump_function_array_mod_2[j,0])+" "+str(bump_function_array_mod_2[j,1])+"\n")
    
    f.write(str(bump_function_array_mod_2[bump_function_array_mod_2.shape[0]-1,0])+" "+"0"+"\n")
    #bump 2 end
    f.write(str(end_x+diameter)+" "+"0"+"\n")
    f.close()
    
    #Normalize bump_center and bump_radius
    bump_radius_1=(bump_radius_1-bump_radius_min)/(bump_radius_max-bump_radius_min)
    bump_radius_2=(bump_radius_2-bump_radius_min)/(bump_radius_max-bump_radius_min)
    bump_radius_3=(bump_radius_3-bump_radius_min)/(bump_radius_max-bump_radius_min)

    return(bump_radius_1,bump_radius_2,bump_radius_3)
#Br1, Br2, Br3 = generate_geometry_start(75,125,175,0,0,0,0,33)

def generate_geometry_train(nr_iter,nr_steps,bump_center_1,bump_center_2,bump_center_3,previous_radius_1,previous_radius_2,previous_radius_3,radius_change_1,radius_change_2,radius_change_3,bump_radius_min,bump_radius_max):

   #parameters
   diameter=25
   start_x=0
   end_x=250
   
   #Denormalize bump_radius
   previous_radius_1=previous_radius_1*(bump_radius_max-bump_radius_min)+bump_radius_min
   previous_radius_2=previous_radius_2*(bump_radius_max-bump_radius_min)+bump_radius_min
   previous_radius_3=previous_radius_3*(bump_radius_max-bump_radius_min)+bump_radius_min

   #Change radius
   bump_radius_1=previous_radius_1+radius_change_1
   bump_radius_2=previous_radius_2+radius_change_2
   bump_radius_3=previous_radius_3+radius_change_3


   death_max1 = False
   death_min1 = False
   death_max2 = False
   death_min2 = False
   death_max3 = False
   death_min3 = False   


   if bump_radius_1 >= bump_radius_max:
      death_max1 = True
   elif bump_radius_2 >= bump_radius_max:
      death_max2 = True
   elif bump_radius_3 >= bump_radius_max:
      death_max3 = True
   elif bump_radius_1 <= bump_radius_min:
      death_min1 = True
   elif bump_radius_2 <= bump_radius_min:
      death_min2 = True
   elif bump_radius_3 <= bump_radius_min:
      death_min3 = True 
      
   else: # death_max1 == False and death_min1 == False and death_max2 == False and death_min2 == False: 
      #List for bump function
      bump_function_list=[]

      #bump function
      for i in range(-99,100):

         bp = np.exp(1/((i/100)**2-1))
         bump_function_list.append([(i/100),bp])

         #strecken und stauchen
         bump_function_array=np.array(bump_function_list)

         bump_function_array_mod_1=bump_function_array
         bump_function_array_mod_1[:,0]=bump_function_array_mod_1[:,0]*bump_radius_1+bump_center_1
         bump_function_array_mod_1[:,1]=(bump_function_array_mod_1[:,1]*-bump_radius_1)+diameter

      #List for bump function
      bump_function_list=[]

      #bump function
      for i in range(-99,100):
         bp = np.exp(1/((i/100)**2-1))
         bump_function_list.append([(i/100),bp])

         #strecken und stauchen
         bump_function_array=np.array(bump_function_list)

         bump_function_array_mod_2=bump_function_array
         bump_function_array_mod_2[:,0]=bump_function_array_mod_2[:,0]*bump_radius_2+bump_center_2
         bump_function_array_mod_2[:,1]=bump_function_array_mod_2[:,1]*bump_radius_2
      #List for bump function
      bump_function_list=[]

      #bump function
      for i in range(-99,100):
         bp = np.exp(1/((i/100)**2-1))
         bump_function_list.append([(i/100),bp])

         #strecken und stauchen
         bump_function_array=np.array(bump_function_list)

         bump_function_array_mod_3=bump_function_array
         bump_function_array_mod_3[:,0]=bump_function_array_mod_3[:,0]*bump_radius_3+bump_center_3
         bump_function_array_mod_3[:,1]=(bump_function_array_mod_3[:,1]*-bump_radius_3)+diameter

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
      f.write(str(3*bump_function_array.shape[0]+3)+"\n")
      f.write("\n")
      f.write(str(end_x+diameter)+" "+"0"+"\n")
      f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
      #bump 1 start 
      #f.write(str(bump_function_array_mod_1[bump_function_array_mod_1.shape[0]-1,0])+" "+"0"+"\n")
      for l in range(bump_function_array.shape[0]-1,1,-1):
         f.write(str(bump_function_array_mod_3[l,0])+" "+str(bump_function_array_mod_3[l,1])+"\n")
      f.write(str(bump_function_array_mod_3[0,0])+" "+str(diameter)+"\n")
      
      for k in range(bump_function_array.shape[0]-1,1,-1):
         f.write(str(bump_function_array_mod_1[k,0])+" "+str(bump_function_array_mod_1[k,1])+"\n")

      f.write(str(bump_function_array_mod_1[0,0])+" "+str(diameter)+"\n")
      #bump 1 end
      f.write("-"+str(diameter)+" "+str(diameter)+"\n")
      f.write("-"+str(diameter)+" "+"0"+"\n")
      #bump 2 start
      f.write(str(bump_function_array_mod_2[0,0])+" "+"0"+"\n")
 
      for j in range(1,bump_function_array.shape[0]-1):
         f.write(str(bump_function_array_mod_2[j,0])+" "+str(bump_function_array_mod_2[j,1])+"\n")

      f.write(str(bump_function_array_mod_2[bump_function_array_mod_2.shape[0]-1,0])+" "+"0"+"\n")
      #bump 2 end
      f.write(str(end_x+diameter)+" "+"0"+"\n")
      f.close()
      #Normalize bump_center and bump_radius
      bump_radius_1=(bump_radius_1-bump_radius_min)/(bump_radius_max-bump_radius_min)
      bump_radius_2=(bump_radius_2-bump_radius_min)/(bump_radius_max-bump_radius_min)
      bump_radius_3=(bump_radius_3-bump_radius_min)/(bump_radius_max-bump_radius_min)

   return bump_radius_1,bump_radius_2,bump_radius_3, death_max1, death_min1, death_max2, death_min2, death_max3, death_min3

