from corsikaio import CorsikaParticleFile
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
jobs_parent_path = "/home/messuti/DataCorsika7"                             # TODO path where to find folders of all jobs
interaction_model = "7631"                                                  # TODO read only one interaction model. Chose one from "7631", "7611", "7231", "7211"
#img_saving_folder = "/home/messuti/DataCorsika7/imgs_xy/"

particle_names =        ["mu_p","mu_a", "vie_p","vie_a","vimu_p","vimu_a"]  # identification names of particles I want to investigate
particle_descriptions = [5000,  6000,   66000,  67000,  68000,   69000]     # code identification of particles in corsika
part_des = particle_descriptions
Npt = len(particle_names)                                                   # Number of Types of Particles investigated

perc = 0.9                                                                  # TODO percentage of particles with lower px (py and x and y) on wich compute std
quantities_of_interest = ["JobID","Primary_energy","Starting_height","Number_of_particles","Cumulative_Energy_of_particles",
                          "abspx_max","abspy_max","std_px_90_perc_particles","std_py_90_perc_particles","log10pzmax","log10pzmin","log10rmax", 
                          "std_x_90_perc_particles", "std_y_90_perc_particles"] + [f"log10r_Q0.{i+1}" for i in range (9)] # Q0.1 è quantile 10%..
qoi = quantities_of_interest

path_output = "/home/messuti/Desktop/"                              # Path to existing folder: where to output dataframes and error files
if path_output[-1] != "/": path_output += "/"
ef = open(f"{path_output}errori_lettura_file.txt","w")
ee = open(f"{path_output}errori_lettura_eventi.txt","w")
ep = open(f"{path_output}errori_lettura_particles.txt","w")
edict = open(f"{path_output}errori_inserimento_valore_key.txt","w") # alcune volte avevo 0 particelle. quindi np.max o sum (non ricordo) dava problemi



# dictionary where to store the info for each particle. will be converted to pandas dataframe
dizio={name:{key:[] for key in qoi} for name in particle_names} 


# Browse list of job folders. Automatically read all jobs in folders nested in "jobs_parent_path"
# Here I retrieve paths inside the jobs_parent_path that show "job" and "em" in the name
job_paths_to_read=[]
with os.scandir(jobs_parent_path) as entries:
    for entry in entries:
        if entry.is_dir() and "job" in str(entry.name) and "em" in str(entry.name) and str(interaction_model) in str(entry.name):
                job_paths_to_read.append(entry.path)

count_entries = {name:0 for name in particle_names}



for job in job_paths_to_read:
    # parsing - it will be something like /path/job##_contver../DAT00## 
    ff = job + f"/DAT{job.split("/")[-1].split("_")[0][3:].zfill(6)}"
    # Opening the selected file
    try:
        with CorsikaParticleFile(ff) as f:
            # selecting a single event
            try:
                for nev,ev in enumerate(f):
                    # selecting all particles of a fixed type, for each type
                    try:
                        for i,pname in enumerate(particle_names):
                            # selecting particles with ID ∈ [part_des[i], part_des[i]+1000)
                            count_entries[pname]  += 1
                            particle_temp = ev.particles[(ev.particles["particle_description"]>=part_des[i]) * (ev.particles["particle_description"]<part_des[i]+1000)] 
                            abs_px = np.abs(particle_temp["px"])    
                            abs_px.sort()                               # select and sort absolute values of px
                            abs_py = np.abs(particle_temp["py"])
                            abs_py.sort()                               # select and sort absolute values of py
                            abs_x = np.abs(particle_temp["x"])
                            abs_x.sort()
                            abs_y = np.abs(particle_temp["y"])
                            abs_y.sort()    
                            particle_perc_px = particle_temp["px"][
                                            (particle_temp["px"] < abs_px[int(perc * len(abs_px)) - 1])
                                            & (-particle_temp["px"] < abs_px[int(perc * len(abs_px)) - 1])] # Select only the perc% particles with lower px
                            particle_perc_py = particle_temp["py"][
                                            (particle_temp["py"] < abs_py[int(perc * len(abs_py)) - 1])
                                            & (-particle_temp["py"] < abs_py[int(perc * len(abs_py)) - 1])]
                            particle_perc_x = particle_temp["x"][
                                            (particle_temp["x"] < abs_x[int(perc * len(abs_x)) - 1])
                                            & (-particle_temp["x"] < abs_x[int(perc * len(abs_x)) - 1])]
                            particle_perc_y = particle_temp["y"][
                                            (particle_temp["y"] < abs_y[int(perc * len(abs_y)) - 1])
                                            & (-particle_temp["y"] < abs_y[int(perc * len(abs_y)) - 1])]
                            
                            dizio[pname]["JobID"].append(f"{job.split("/")[-1]}_event{nev}")
                            dizio[pname]["Primary_energy"].append(ev.header["total_energy"])
                            dizio[pname]["Starting_height"].append(ev.header["starting_height"])    # TODO it does not work. It gives always 11282920.0
                            dizio[pname]["Number_of_particles"].append(len(particle_temp))
                            dizio[pname]["Cumulative_Energy_of_particles"].append(np.sqrt((particle_temp["px"]**2+particle_temp["py"]**2+particle_temp["pz"]**2).sum()))
                            dizio[pname]["abspx_max"].append(np.max(np.abs(particle_temp["px"])))
                            dizio[pname]["std_px_90_perc_particles"].append(np.std(particle_perc_px)) # measure the dispersion of particles (outliers excluded)
                            dizio[pname]["abspy_max"].append(np.max(np.abs(particle_temp["py"])))
                            dizio[pname]["std_py_90_perc_particles"].append(np.std(particle_perc_py)) # measure the dispersion of particles (outliers excluded)
                            dizio[pname]["log10pzmax"].append(np.log10(np.max(particle_temp["pz"])))
                            dizio[pname]["log10pzmin"].append(np.log10(np.min(particle_temp["pz"])))
                            log10r = np.log10(np.sqrt(particle_temp["x"]**2+particle_temp["y"]**2))
                            dizio[pname]["log10rmax"].append(np.max(log10r))
                            dizio[pname]["std_x_90_perc_particles"].append(np.std(particle_perc_x)) # measure the dispersion of particles (outliers excluded)
                            dizio[pname]["std_y_90_perc_particles"].append(np.std(particle_perc_y)) # measure the dispersion of particles (outliers excluded)
                            for __ in range(9):
                                dizio[pname][f"log10r_Q0.{__+1}"].append(np.quantile(log10r, (__+1)/10))


                            # TODO uncomment for some plot (plot_x_y or plot_px_pz or other)
                            # if nev > 200: break #i do not want too many plots
                            # try:
                            #     jobi = ff.split("/")[-2]
                            #     energy = ev.header["total_energy"]
                            #     if pname=="mu_p": plot_x_y(particle_temp,img_saving_folder,energy,jobi,nev,pname)
                            # except Exception as mai:
                            #     print("PLOT error", mai)

                            
                            for key in dizio[pname]:
                                #print(len(dizio[pname][key]),"\t", count_entries[pname])
                                if len(dizio[pname][key])!=count_entries[pname]:
                                    # sometimes, for lower energies I founded vie 0 particles. This gave me problems when buildinf pandas dataframes. I fill with nans
                                    dizio[pname][key].append(np.nan)
                                    edict.write(f"inserted nan here\t{ff}\tevent_{nev}\tparticle_{pname}\t{key}\n\n")
                    except Exception as exp:
                        # occurent exception in reading particle
                        ep.write(f"{str(exp)}\t{ff}\tevent_{nev}\tparticle_{pname}\n\n")
            except Exception as exe:
                # occurred exception in reading event
                ee.write(f"{str(exe)}\t{ff}\tevent_{nev}\n\n")     
    except Exception as exf:
        # encountered exception in reading file
        ef.write(f"{str(exf)}\t{ff}\n\n")
ef.close()
ee.close()
ep.close()
edict.close()
    
for particle in dizio:
    df = pd.DataFrame.from_dict(dizio[particle])
    df.to_csv(f"{path_output}{particle}_{interaction_model}.csv",index=False)