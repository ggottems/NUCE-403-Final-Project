import openmc

def TRISOp(radii,materials):
    ''' '''
    spheres=[]
    for i in range(len(radii)):
        sphere=openmc.Sphere(r=radii[i])
        spheres.append(sphere)
    cells=[]
    for i in range(len(spheres)):
        if i==0:
            cell_i=openmc.Cell(fill=materials[i],region=-spheres[i])
        else:
            cell_i=openmc.Cell(fill=materials[i],region=+spheres[i-1]&-spheres[i])
        cells.append(cell_i)
    triso_universe = openmc.Universe(cells=cells)
    return triso_universe
