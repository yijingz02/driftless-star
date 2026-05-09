#Initialization
import os
current_path = os.path.dirname(os.path.realpath(__name__))
import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'gpu')
print(jax.devices()) # TFRT_CPU_0
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import h5py as h5
import interpax
import matplotlib.pyplot as plt
import NEOPAX
from NEOPAX._constants import elementary_charge



Input_Path=os.path.join(current_path,'./examples/inputs/')
Output_Path=os.path.join(current_path,'./examples/Solve_Er_General/')



vmec_file=os.path.join(Input_Path,'wout_HSX_QHS_vacuum_ns201.nc')
boozer_file=os.path.join(Input_Path, 'boozmn_HSX_QHS_vacuum_ns201.nc')
neoclassical_file=os.path.join(Input_Path,'Dij_HSX_QHS_vacuum_ns201.h5')
profiles_file=os.path.join(Input_Path,'Impurity_SFINCS_mixed.h5')
neoclassical_option= 1


file_profiles_h5=h5.File(profiles_file,'r')
r_file=file_profiles_h5['r'][()]
Te_file=file_profiles_h5['Te'][()]
TD_file=file_profiles_h5['TD'][()]
TT_file=file_profiles_h5['TT'][()]
ne_file=file_profiles_h5['ne'][()]
nD_file=file_profiles_h5['nD'][()]
nT_file=file_profiles_h5['nT'][()]
file_profiles_h5.close()
#Solver parameters 
t0=0.0            #Initial time
t_final=20.       #Final time
dt=0.0001         #Initial guess of timestep
ts_list=jnp.linspace(t0,t_final,10)       #Times at which to save the solution
rtol=1.e-5        #Tolerance for solver
atol=1.e-5        #Tolerance for solver
momentum_correction_flag=False   #Consider momementum correection (True) or not (False)
DEr=2.0           #Electric field diffusion (m^2/s)
Er_relax=0.1      #Relaxation time of electric field equation
on_Er=1           #Electric field equation on (1), off (0)
on_Pe=0           #Electron energy equation on (1), off (0)
on_PD=0           #Deuterium energy equation on (1), off (0)
on_PT=0           #Tritium energy equation on (1), off (0)
chi=jnp.ones(3)*0.0065   #Anomalous heat transport coefficients for e,D,T 
 
#Create parameters class
parameters=NEOPAX.Solver_Parameters(t0=t0,t_final=t_final,dt=dt,ts_list=ts_list,rtol=rtol,
                        atol=atol,momentum_correction_flag=momentum_correction_flag,DEr=DEr,Er_relax=Er_relax,
                        on_Er=on_Er,on_ne=None,on_nD=None,on_nT=None,on_nHe=None,on_Pe=on_Pe,on_PD=on_PD,on_PT=on_PT,
                        chi=chi,on_OmegaC=None)
        



#Create grid
n_species=3
Nx=4
n_radial=41
grid=NEOPAX.Grid.create_standard(n_radial,Nx,n_species)

#Get magnetic configuration related quantities from class Field
field=NEOPAX.Field.read_vmec_booz(n_radial,vmec_file,boozer_file)

#Defining species data
ne0 = 4.21e20
te0 = 17.8e+3
ni0 = 4.21e20
ti0 = 17.8e+3
neb = 0.6e20
teb = 0.7e+3
nib = 0.6e20
tib = 0.7e+3

n_scale=1.#0.65       
T_scale=1.#0.7       
deuterium_ratio=0.5    
tritium_ratio=0.5  

#Boundaries for temperature and density (to be improved)
#T_edge=jnp.array([0.7*1.e+3,0.7*1.e+3,0.7*1.e+3])
#n_edge=jnp.array([0.6e+20,deuterium_ratio*0.6e+20,tritium_ratio*0.6e+20])

#Define species profiles 
#Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/field.a_b)**2)+teb)
#ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/field.a_b)**10.)+neb)
#TD_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
#nD_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)
#TT_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
#nT_initial_function = lambda r: n_scale*tritium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)
#Te_initial_function= lambda r: T_scale*((te0-teb)*(1-(r/field.a_b)**2)+teb)
#ne_initial_function =lambda r: n_scale*((ne0-neb)*(1-(r/field.a_b)**10.)+neb)
#TD_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
#nD_initial_function = lambda r: n_scale*deuterium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)
#TT_initial_function= lambda r: T_scale*((ti0-tib)*(1-(r/field.a_b)**2)+tib)
#nT_initial_function = lambda r: n_scale*tritium_ratio*((ni0-nib)*(1-(r/field.a_b)**10.)+nib)

Er_initial_function = lambda x: 100.*x*(0.8-x)

#Er0 = SpatialDiscretisation.discretise_fn(r0, r_final, n_radial, Er_initial_function)

#Initialize species
#Te_initial=Te_initial_function(field.r_grid)
#ne_initial=ne_initial_function(field.r_grid)
#TD_initial=TD_initial_function(field.r_grid)
#nD_initial=nD_initial_function(field.r_grid)
#TT_initial=TT_initial_function(field.r_grid)
#nT_initial=nT_initial_function(field.r_grid)
Er_initial=Er_initial_function(field.rho_grid)


temperature_initial=jnp.zeros((n_species,n_radial))
density_initial=jnp.zeros((n_species,n_radial))
temperature_initial=temperature_initial.at[0,:].set(Te_file*1.e+3)
temperature_initial=temperature_initial.at[1,:].set(TD_file*1.e+3)
temperature_initial=temperature_initial.at[2,:].set(TT_file*1.e+3)
density_initial=density_initial.at[0,:].set(ne_file*1.e+20)
density_initial=density_initial.at[1,:].set(nD_file*1.e+20)
density_initial=density_initial.at[2,:].set(nT_file*1.e+20)

T_edge=jnp.array([temperature_initial[0,-1],temperature_initial[1,-1],temperature_initial[2,-1]])
n_edge=jnp.array([density_initial[0,-1],density_initial[1,-1],density_initial[2,-1]])

fr=0.0
temperature_initial=temperature_initial.at[:,0].set((4.*temperature_initial.at[:,1].get()-temperature_initial.at[:,2].get()-fr*2*field.dr)/(3.0))   
temperature_initial=temperature_initial.at[:,-1].set(T_edge)
density_initial=density_initial.at[:,0].set((4.*density_initial.at[:,1].get()-density_initial.at[:,2].get()-fr*2*field.dr)/(3.0))   
density_initial=density_initial.at[:,-1].set(n_edge)
mass=jnp.array([1 / 1836.15267343,2,3])
charge=jnp.array([-1,1,1])

#Initialçize species
Global_species=NEOPAX.Species(n_species,n_radial,grid.species_indeces,mass,charge,temperature_initial,density_initial,Er_initial,field.r_grid,field.r_grid_half,field.dr,field.Vprime_half, field.overVprime,n_edge,T_edge)


#Read monoenergetic database, from MONKES 
database=NEOPAX.Monoenergetic.read_monkes(field.a_b,neoclassical_file)

turbulent=NEOPAX.Turbulence.from_analytical_model(Global_species,jnp.zeros(3))

#Initial profiles
ne0=Global_species.density[0]*1.e-20
nD0=Global_species.density[1]*1.e-20
nT0=Global_species.density[2]*1.e-20
Pe0=Global_species.temperature[0]*1.e-3*ne0
PD0=Global_species.temperature[1]*1.e-3*nD0
PT0=Global_species.temperature[2]*1.e-3*nT0
Er0=Global_species.Er

#Define initial coupled solutions to diffrax (can solve tupples of spatial descritizations!!! already tested)
y0=(Er0,Pe0,PD0,PT0,ne0,nD0,nT0)#,ne_initial,nT_initial,nD_initial,nHe_initial in r=rho*a_b!!!!
args=Global_species,grid,field,database,turbulent,parameters


sol=NEOPAX.solve_transport_equations(y0,args)

final_species=NEOPAX.Species(n_species,n_radial,grid.species_indeces,mass,charge,temperature_initial,density_initial,sol.ys[0][-1,:],field.r_grid,field.r_grid_half,field.dr,field.Vprime_half, field.overVprime,n_edge,T_edge)
Gamma_mom,Q_mom,Upar_mom,qpar_mom,Upar2_mom=NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(final_species,grid,field,database)

J_boots=(-Upar_mom[:,0]+Upar_mom[:,1]+Upar_mom[:,2])*elementary_charge
Er_final=sol.ys[0][-1,:]

file=h5.File('NEOPAX_output.h5','w')
file['rho']=field.rho_grid
file['Er']=Er_final
file['Jboots']=J_boots
file.close()

