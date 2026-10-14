# -------------------------
#  KAIROS POWER PEBBLE BED
# -------------------------

import openmc

xs_path = '/home/jpontes/xs_libraries/endfb-viii.0-hdf5/cross_sections.xml' # path to cross section library

# TRISO

kernel_radius = 212.5e-4
buffer_thickness = 100e-4
in_pyc_thickness = 40e-4
sic_thickness = 35e-4
out_pyc_thickness = 40e-4

TRISO_radii = [kernel_radius, 
               kernel_radius+buffer_thickness, 
               kernel_radius+buffer_thickness+in_pyc_thickness, 
               kernel_radius+buffer_thickness+in_pyc_thickness+sic_thickness, 
               kernel_radius+buffer_thickness+in_pyc_thickness+sic_thickness+out_pyc_thickness]
fuel_enrichment = 19.75

# PEBBLE
inner_graphite_radius = 2.07 #cm
fuel_region_radius = 2.7 #cm 
outer_graphite_radius = 3 #cm  
pebble_diameter = 6 # cm

# CORE 

core_cylinder_radius = 130 #
core_outer_radius = 150 
core_height = 200 #84 #cm

#    MATERIALS

T_avg = 620+273.15
materials_list = []

#Kernel
UO2 = openmc.Material(name='UO2')
UO2.add_element('U', 1, enrichment=fuel_enrichment)
UO2.add_element('O', 2)
UO2.set_density('g/cm3', 10.4)
UO2.temperature = T_avg
materials_list.append(UO2)

Buffer = openmc.Material(name='Buffer')
Buffer.add_element('C', 1)
Buffer.set_density('g/cm3', 1.0)
Buffer.temperature = T_avg
#Buffer.add_s_alpha_beta('c_Graphite')
materials_list.append(Buffer)

PyC = openmc.Material(name='PyC')
PyC.add_element('C', 1) #pyrolitic carbon
PyC.set_density('g/cm3', 1.9)
PyC.temperature = T_avg
materials_list.append(PyC)

SiC = openmc.Material(name='SiC')
SiC.add_element('C', 1)
SiC.add_element('Si', 1)
SiC.set_density('g/cm3', 3.2)
SiC.temperature = T_avg
materials_list.append(SiC)

#Monolith
graphite_compact = openmc.Material(name='Graphite Compact')
graphite_compact.add_element('C', 1)
graphite_compact.set_density('g/cm3', 1.74)
graphite_compact.temperature = T_avg
materials_list.append(graphite_compact)

graphite = openmc.Material(name='Graphite')
graphite.add_element('C', 1)
graphite.set_density('g/cm3', 1.74)
graphite.temperature = T_avg
graphite.add_s_alpha_beta('c_Graphite') #for crystalline graphite;;
materials_list.append(graphite)

graphite_low = openmc.Material(name='Low-Density Graphite')
graphite_low.add_element('C', 1)
graphite_low.set_density('g/cm3', 1.41)
graphite_low.temperature = T_avg
graphite_low.add_s_alpha_beta('c_Graphite') #for crystalline graphite;;
materials_list.append(graphite_low)

#FLiBe
'''
LiF = openmc.Material(name = 'LiF')
LiF.add_element('Li', 1, 'ao')
LiF.add_element('F', 1, 'ao')
#LiF.set_density('g/cm3', )
LiF.temperature = T_avg

BeF2 = openmc.Material(name = 'BeF2')
BeF2.add_element('Be', 1, 'ao')
BeF2.add_element('F', 2, 'ao')
#BeF2.set_density('g/cm3', )
BeF2.temperature = T_avg'''

#FLiBe = openmc.Material.mix_materials([LiF, BeF2], [2/3, 1/3], 'ao', name = 'FLiBe')
#FLiBe.set_density('g/cm3', 1.952) # @ 600-700C
#materials_list.append(FLiBe)

FLiBe = openmc.Material(name="FLiBe")
for isotope, number_density in {
    "Li6": 4.38333e-7,
    "Li7": 2.40010e-2,
    "Be9": 1.20001e-2,
    "F19": 4.80084e-2}.items():
    FLiBe.add_nuclide(isotope, number_density)
FLiBe.set_density("sum") 
FLiBe.temperature = T_avg
materials_list.append(FLiBe)
import json

with open('volume_fraction_dict.json', 'r') as f:
    volume_fractions = json.load(f)

triso_homog = openmc.Material.mix_materials([UO2,Buffer,PyC,SiC,PyC], volume_fractions, 'vo',name='TRISO homog')

pebble_homog = openmc.Material.mix_materials([triso_homog,graphite_compact],[0.3,0.7],'vo',name='Homogenized Pebble')
pebble_homog.temperature = T_avg
#pebble_homog.set_density('g/cm3', 4.912)

materials_list.append(pebble_homog)

'''import json

with open('/Users/jxd6218/Documents/OpenMC/gHPMR/volume_fraction_dict.json', 'r') as f:
    volume_fraction_dict2 = json.load(f)

triso_homog = openmc.Material.mix_materials([UO2,Buffer,PyC,SiC,PyC],[volume_fraction_dict2['UO2'],volume_fraction_dict2['Buffer'],
                                                                    volume_fraction_dict2['PyC'],volume_fraction_dict2['SiC'],
                                                                    volume_fraction_dict2['PyC']], 'vo',name='TRISO homog')

pebble_homog = openmc.Material.mix_materials([triso_homog,graphite_compact],[0.2843,0.7157],'vo',name='Homogenized Pebble')
#pebble_homog.temperature = 1150

materials_list.append(pebble_homog)'''