import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def generate_geometry_start(bump_center,bump_radius,bump_radius_min,bump_radius_max):

   #parameters
   diameter=20
   start_x=0
   end_x=200

   #List for bump function
   bump_function_list=[]

   #bump function
   for i in range(-99,100):
 
      bp = np.exp(1/((i/100)**2-1))
      bump_function_list.append([(i/100),bp])

      #strecken und stauchen
      bump_function_array=np.array(bump_function_list)

      bump_function_array_mod=bump_function_array
      bump_function_array_mod[:,0]=bump_function_array_mod[:,0]*bump_radius+bump_center
      bump_function_array_mod[:,1]=bump_function_array_mod[:,1]*bump_radius
   
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
   f.write(str(bump_function_array.shape[0]+5)+"\n")
   f.write("\n")
   f.write(str(end_x+diameter)+" "+"0"+"\n")
   f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
   f.write("-"+str(diameter)+" "+str(diameter)+"\n")
   f.write("-"+str(diameter)+" "+"0"+"\n")
   f.write(str(bump_function_array_mod[0,0])+" "+"0"+"\n")
   
   for j in range(1,bump_function_array.shape[0]-1):
      f.write(str(bump_function_array_mod[j,0])+" "+str(bump_function_array_mod[j,1])+"\n")
   
   f.write(str(bump_function_array_mod[bump_function_array_mod.shape[0]-1,0])+" "+"0"+"\n")
   f.write(str(end_x+diameter)+" "+"0"+"\n")
   f.close()

   #Normalize bump_center and bump_radius
   bump_radius=(bump_radius-bump_radius_min)/(bump_radius_max-bump_radius_min)

   return(bump_radius)

def generate_geometry_init(nr_iter,nr_steps,bump_center,bump_radius_min,bump_radius_max):

   #parameters
   diameter=20
   start_x=0
   end_x=200

   #bump radius
   bump_radius=np.random.randint(bump_radius_min,bump_radius_max)

   #List for bump function
   bump_function_list=[]

   #bump function
   for i in range(-99,100):
 
      bp = np.exp(1/((i/100)**2-1))
      bump_function_list.append([(i/100),bp])

      #strecken und stauchen
      bump_function_array=np.array(bump_function_list)

      bump_function_array_mod=bump_function_array
      bump_function_array_mod[:,0]=bump_function_array_mod[:,0]*bump_radius+bump_center
      bump_function_array_mod[:,1]=bump_function_array_mod[:,1]*bump_radius
   
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
   f.write(str(bump_function_array.shape[0]+5)+"\n")
   f.write("\n")
   f.write(str(end_x+diameter)+" "+"0"+"\n")
   f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
   f.write("-"+str(diameter)+" "+str(diameter)+"\n")
   f.write("-"+str(diameter)+" "+"0"+"\n")
   f.write(str(bump_function_array_mod[0,0])+" "+"0"+"\n")
   
   for j in range(1,bump_function_array.shape[0]-1):
      f.write(str(bump_function_array_mod[j,0])+" "+str(bump_function_array_mod[j,1])+"\n")
   
   f.write(str(bump_function_array_mod[bump_function_array_mod.shape[0]-1,0])+" "+"0"+"\n")
   f.write(str(end_x+diameter)+" "+"0"+"\n")
   f.close()

   #Write image to stats
   plt.figure(figsize=(20, 2))
   plt.plot(bump_function_array_mod[:,0],bump_function_array_mod[:,1])
   plt.xlim(0,200)
   plt.ylim(0,20)
   plt.xticks([], [])
   plt.yticks([], [])
   #plt.title("Iteration_"+"%06d"%nr_iter+"_Step_"+"%06d"%nr_steps+"_center_"+"%06d"%bump_center+"_radius_"+"%06d"%bump_radius)
   #plt.savefig("stats/geo_"+"%06d"%nr_iter+"_"+"%06d"%nr_steps+"_center_"+"%06d"%bump_center+"_radius_"+"%06d"%bump_radius+".png")
   plt.close()

   #Normalize bump_center and bump_radius
   bump_radius=(bump_radius-bump_radius_min)/(bump_radius_max-bump_radius_min)

   return(bump_radius)

def generate_geometry_train(nr_iter,nr_steps,bump_center,previous_radius,radius_change,bump_radius_min,bump_radius_max):

   #parameters
   diameter=20
   start_x=0
   end_x=200
   
   #Denormalize bump_radius
   previous_radius=previous_radius*(bump_radius_max-bump_radius_min)+bump_radius_min

   #Change radius
   bump_radius=previous_radius+radius_change

   #Flag for agent death
   death=False

   #Keep max and min bounds of bump radius
   if bump_radius >= bump_radius_max:
       death=True

   if bump_radius <= bump_radius_min:
       death=True

   #List for bump function
   bump_function_list=[]

   #bump function
   for i in range(-99,100):
 
      bp = np.exp(1/((i/100)**2-1))
      bump_function_list.append([(i/100),bp])

      #strecken und stauchen
      bump_function_array=np.array(bump_function_list)

      bump_function_array_mod=bump_function_array
      bump_function_array_mod[:,0]=bump_function_array_mod[:,0]*bump_radius+bump_center
      bump_function_array_mod[:,1]=bump_function_array_mod[:,1]*bump_radius

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
   f.write(str(bump_function_array.shape[0]+5)+"\n")
   f.write("\n")
   f.write(str(end_x+diameter)+" "+"0"+"\n")
   f.write(str(end_x+diameter)+" "+str(diameter)+"\n")
   f.write("-"+str(diameter)+" "+str(diameter)+"\n")
   f.write("-"+str(diameter)+" "+"0"+"\n")
   f.write(str(bump_function_array_mod[0,0])+" "+"0"+"\n")
   
   for j in range(1,bump_function_array.shape[0]-1):
      f.write(str(bump_function_array_mod[j,0])+" "+str(bump_function_array_mod[j,1])+"\n")
   
   f.write(str(bump_function_array_mod[bump_function_array_mod.shape[0]-1,0])+" "+"0"+"\n")
   f.write(str(end_x+diameter)+" "+"0"+"\n")
   f.close()

   #Write image to stats
   plt.figure(figsize=(20, 2))
   plt.plot(bump_function_array_mod[:,0],bump_function_array_mod[:,1])
   plt.xlim(0,200)
   plt.ylim(0,20)
   plt.xticks([], [])
   plt.yticks([], [])
   plt.title("Iteration_"+"%06d"%nr_iter+"_Step_"+"%06d"%nr_steps+"_center_"+"%06d"%bump_center+"_radius_"+"%06d"%bump_radius)
   #plt.savefig("stats/geo_"+"%06d"%nr_iter+"_"+"%06d"%nr_steps+"_center_"+"%06d"%bump_center+"_radius_"+"%06d"%bump_radius+".png")
   plt.close()

   #Normalize bump_center and bump_radius
   bump_radius=(bump_radius-bump_radius_min)/(bump_radius_max-bump_radius_min)
   
   return(bump_radius,death)
