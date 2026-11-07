"""Module contains functions related to completing an inp deck outputted from GMSH so it can be used
by Calculix ccx."""

import numpy as np
import src.environment.stimulus as stim
from src.devo.geometry import Point
from src.devo.topology import Edge


def _write_shell_elements(file):

    lines = [line.strip() for line in file if not line.isspace()]

    for ind, line in enumerate(lines):
        if line == "******* E L E M E N T S *************":
            starter = ind+1
            elements = lines[ind:]

    indices = []
    for ind, line in enumerate(lines):
        if line.split(" ")[0] == "*ELEMENT,":
            marker = line.split(" ")[-1].split("=")[-1]
            marker = ''.join(filter(lambda x: not x.isdigit(), marker))
            if marker == "Surface":
                indices.append(ind)
                breaker = ind
                break
            elif marker == "Volume":
                indices.append(ind)
                break
        elif ind == len(elements)-1:
            indices.append(len(elements))

    del lines[starter:breaker]

    file.truncate(0)
    file.seek(0)
    for each_line in lines:
        if each_line.split(" ")[0] == "*ELEMENT,":
            #each_line.replace("type=CPS3,", "type=S3,")("type=CPS3,", "type=S3,")
            each_line = str(each_line).replace("type=CPS3,", "type=S3,")
            file.write(str(each_line)+"\n")
        else:
            file.write(str(each_line)+"\n")


def _write_stimuli_groups(file, stimuli_entities):

    stimulus_script = ""
    for stimulus, entity in stimuli_entities.items():

        if isinstance(stimulus, stim.FixedSupport):
            stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{entity[0] + 1}\n"
        elif isinstance(stimulus, stim.RollerSupport) or isinstance(stimulus, stim.PinnedSupport):
            stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{entity[0]  + 1}\n"
        elif isinstance(stimulus, stim.PointLoad):
            stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{entity[0]  + 1}\n"
        elif isinstance(stimulus, stim.Temperature):
            for ent in entity:
                if isinstance(ent, Point):
                    stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{ent.id + 1}\n"
                elif isinstance(ent, Edge):
                    nodes = [ent.start_point.id+1, ent.end_point.id+1]
                    stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{nodes[0]},{nodes[1]}\n"
        elif isinstance(stimulus, stim.PlanarDistributedLoad):
            stimulus_script += f"*ELSET, ELSET={stimulus.group_name}\n"
            if stimulus.application_type == "planar_face_DLOAD":
                if len(entity) > 16:
                    lines = [entity[i:i+16] for i in range(0, len(entity), 16)]
                    for ent in lines:
                        ent= np.asarray(ent)
                        stimulus_script += np.array2string(ent, separator=', ').strip("[]") +'\n'
                else:
                    for ent in entity:
                        stimulus_script += f"{ent},"
                    stimulus_script += "\n"
        elif isinstance(stimulus, stim.FixedSupportPlanar):
            
            stimulus_script += f"*NSET, NSET={stimulus.group_name}\n"
            if len(entity) > 7:
                lines = np.array_split(entity, 7)
                for ent in lines:
                    stimulus_script += np.array2string(ent, separator=', ').strip("[]") +'\n'
            else:
                for ent in entity:
                    stimulus_script += f"{ent},"
                stimulus_script += "\n"            
                
    file.write(stimulus_script)


def _write_generate_node_set(file, node_data, nset_name=None):
    """Write node set for all points in inp.

    Args:
        file: Inp file.
        node_data: Node labels from GMSH.

    Returns:
        None
    """
    if nset_name is None:
        inp_deck = (
            f'*NSET, NSET=ALLNODES, GENERATE\n'
            f'{min(node_data)}, {max(node_data)}, 1\n'
        )
    else:
        inp_deck = (f'*NSET, NSET={nset_name}\n')
        #node_labels = np.array_split(np.array(list(node_data.keys())),1)
        lst = list(node_data.keys())
        node_labels = [np.array(lst[i:i + 5]) for i in range(0, len(list(lst)), 5)]
        for label in node_labels:
            inp_deck += np.array2string(label, separator=', ').strip("[]").strip(" ") +'\n'
    
    file.write(inp_deck)


def _write_generate_element_set(file, element_labels, elset_name=None):
    """Write element set for all elements in inp.

    Args:
        file: Inp file.
        element_labels: Element labels from GMSH.

    Returns:
        None
    """
    if elset_name is None:
        inp_deck = (
            f'*ELSET, ELSET=ALLELEMENTS, GENERATE\n'
            f'{min(element_labels)}, {max(element_labels)}, 1\n'
        )
    else:
        inp_deck = (f'*ELSET, ELSET={elset_name}\n')
        lst = list(element_labels.keys())
        ele_labels = [np.array(lst[i:i + 5]) for i in range(0, len(list(lst)), 5)]
        for label in ele_labels:
            inp_deck += np.array2string(label, separator=', ').strip("[]") +'\n'

    file.write(inp_deck)

def _write_surface(file, name, elset, surface_name):

    inp_deck = ''
    inp_deck += f'*SURFACE, NAME={name}\n{elset}, {surface_name}\n'
    file.write(inp_deck)

def _write_tie_constraint(file, name, surf_1, surf_2):

    inp_deck = ''
    inp_deck += f'*TIE, name={name}, position tolerance=0.01\n{surf_1}, {surf_2}\n'
    file.write(inp_deck)


def _write_beam_sections(file, element_labels, organism, shell_thickness=0.0, materials=None):
    """Write beam section for each edge in inp.

    Args:
        file: Inp file.
        element_labels: Element labels from organism beam mesh property.
        organism: Organism object.

    Returns:
        Dictionaries of materials in organism. Key is material name and value is material object.
    """
    if materials is None:
        materials = {}
    edges = organism.edges
    edge_el_ids = {}
    for edge in edges.values():
        edge_el_ids[edge._el_id] = edge.id

    inp_deck = ''

    for label in element_labels:

        edge_id = edge_el_ids[label]

        material = edges[edge_id].material

        if edges[edge_id].cs_type.value == 0:
            cs_area = edges[edge_id].cs_area

            if cs_area == Edge.min_cs_area:
                pass
            else:
                solid_section = (
                    f'*BEAM SECTION, ELSET=BEAM{label}, MATERIAL="{material.name}", SECTION=CIRC\n'
                    f'{2*np.sqrt(cs_area/np.pi)}\n'
                )
                inp_deck += solid_section

            if material.name not in materials:
                materials[material.name] = material                     

        elif edges[edge_id].cs_type.value == 1:

            if edges[edge_id].dim_1 != 0.0 and edges[edge_id].dim_2 != 0.0:
                dim_1 = edges[edge_id].dim_1
                dim_2 = edges[edge_id].dim_2
                offset = ((dim_1/2) + (shell_thickness/2)) / dim_1
                unit_edge_normal = np.ndarray.round(edges[edge_id]._normal / np.linalg.norm(edges[edge_id]._normal), 6)
                solid_section = (
                    f'*BEAM SECTION, ELSET=BEAM{label}, OFFSET1=-{offset}, MATERIAL="{material.name}", SECTION=RECT\n'
                    f'{dim_1}, {dim_2}\n'
                    f'{unit_edge_normal[0]}, {unit_edge_normal[1]}, {unit_edge_normal[2]}\n'
                )
                inp_deck += solid_section
                if material.name not in materials:
                    materials[material.name] = material                 

    file.write(inp_deck)

    return materials


def _write_plate_shell_section(file, organism, elset_name=None, materials=None):

    if materials is None:
        materials = {}
    inp_deck = ''

    if elset_name is None:
        faces = organism.faces.values()
        for face in faces:
            shell_section = (
                f'*SHELL SECTION, ELSET=Surface{face.id}, MATERIAL="{face.material.name}"\n'
                f'{round(face.thickness, 2)}\n'
            )
            inp_deck += shell_section

            if face.material.name not in materials:
                materials[face.material.name] = face.material
    else:
        face_material = organism.faces[0].material
        face_thickness = organism.faces[0].thickness
        shell_section = (
            f'*SHELL SECTION, ELSET={elset_name}, MATERIAL="{face_material.name}"\n'
            f'{round(face_thickness, 2)}\n'
        )
        inp_deck += shell_section

        if face_material.name not in materials:
            materials[face_material.name] = face_material

    file.write(inp_deck)

    return materials


def _write_materials(file, materials):
    """Write material section of inp.

    Args:
        file: Inp file.
        materials: A dictionary of the materials in the organism.
        Key is material name and value is material object.

    Returns:
        None
    """
    inp_deck = ''
    for name, material in materials.items():
        material_deck = (
            f'*MATERIAL, NAME="{name}"\n'
            f'*DENSITY\n'
            f'{material.density}\n'
            f'*ELASTIC\n'
            f'{material.youngs_mod}, {material.poisson_ratio}\n'
            f'*CONDUCTIVITY, TYPE=ISO\n'
            f'{material.conductivity[0]}, {material.conductivity[1]}\n'
            f'*SPECIFIC HEAT\n'
            f'{material.specific_heat}\n'
            f'*EXPANSION, ZERO={material.expansion[1]}\n'
            f'{material.expansion[0]}\n'
        )
        inp_deck += material_deck
    file.write(inp_deck)


def _write_initial_conditions(file, stimuli_entities):
    """Write initial conditions stimuli into inp.
    If stimulus exists in environment, write it, else assume initial condition temperature.

    Args:
        file: Inp file.
        stimuli: List of environmental stimuli

    Returns:
        None
    """
    inp_deck = ''
    found = False
    for stimulus in stimuli_entities.keys():
        if isinstance(stimulus, stim.InitialTemperature):
            inp_deck += stimulus.get_inp_deck()
            found = True
    if not found:
        inp_deck += '*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALLNODES, 273.15\n'

    file.write(inp_deck)


def _write_temp_from_csv_file(file, temp_file):
    """Write temperature data from frd_file

    Assuming calculix can select from the ASCI frd file the requested data

    Args:
        file: inp file
        temp_file: csv file with node temperatures

    Returns:
        None
    """

    inp_deck = '*TEMPERATURE\n'
    with open(temp_file, 'r') as nt_file:
        for line in nt_file:
            inp_deck += str(line)
        file.write(inp_deck)


def _write_step(file):
    """Write step section of inp.

    - This current configuration will mean a static step is used.

    Args:
        file: Inp file.

    Returns:
        None
    """
    inp_deck = (
        '*STEP\n'
        '*STATIC\n'
        '1.0, 1.0, 1e-05, 1.0\n'
    )
    file.write(inp_deck)

def _write_buckle_step(file):
    """Write buckle step section of inp.

    - This current configuration will mean a buckle step is used.

    Args:
        file: Inp file.

    Returns:
        None
    """
    inp_deck = (
        '*STEP\n'
        '*BUCKLE\n'
        '4\n'
    )
    file.write(inp_deck)


def _write_stimuli(file, stimuli_entities, dim):
    """Write stimuli into inp.

    Args:
        file: Inp file.
        stimuli_enntities: Dictionary associating environmental stimuli with points in organism.
        Key is Point object, value is list of associated stimuli.
        dim: model dimension to determine d.o.f at nodes of elements
    Returns:
        None
    """
    inp_deck = ''

    disp_stim = []

    for stimulus, entities in stimuli_entities.items():
        if (isinstance(stimulus, stim.FixedSupport) or isinstance(stimulus, stim.TwoDimConstraint)
                or isinstance(stimulus, stim.PointLoad) or isinstance(stimulus, stim.PinnedSupport)
                or isinstance(stimulus, stim.RollerSupport)):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.PlanarDisplacement):
            if stimulus.group_name not in disp_stim:
                inp_deck += stimulus.get_inp_deck()
                disp_stim.append(stimulus.group_name)
        elif isinstance(stimulus, stim.FixedSupportPlanar):
            if dim == 1:
                inp_deck += stimulus.get_inp_deck('edge')
            elif dim == 2:
                inp_deck += stimulus.get_inp_deck('face')
        elif isinstance(stimulus, stim.PlanarDistributedLoad):
            inp_deck += stimulus.get_inp_deck(entities[-1])
        elif isinstance(stimulus, stim.PlanarConcentratedLoad):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.Temperature):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.DistributedHeatFlux):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.ConcentratedHeatFlux):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.TranslationBC):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.RotationBC):
            inp_deck += stimulus.get_inp_deck()            
        elif isinstance(stimulus, stim.EncastreBC):
            inp_deck += stimulus.get_inp_deck()  
    file.write(inp_deck)

def _write_stimulus_groups(file, stimulus_entities):
    """
    Write the physical groups (node or element set) from the stimulus
    entities and organism data by matching stimulus with node/element
    id in organism
    """
    
    for stimulus, entity_list in stimulus_entities.items():
        if isinstance(stimulus, stim.FixedSupportPlanar):
            inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
            node_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]
            for ent in node_labels:
                inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'                               
        
        elif isinstance(stimulus, stim.PlanarDistributedLoad):
            inp_deck = f'*ELSET, ELSET={stimulus.group_name}\n'
            if entity_list[0] == "ALLELEMENTS":
                pass
            else:
                entity_list = entity_list[:-1]
                ele_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]
                for ent in ele_labels:
                    inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'

        elif isinstance(stimulus, stim.TranslationBC):
            inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
            node_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]
            for ent in node_labels:
                inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'

        elif isinstance(stimulus, stim.RotationBC):
            inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
            node_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]
            for ent in node_labels:
                inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'

        elif isinstance(stimulus, stim.PointLoad):
            inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
            node_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]               
            for ent in node_labels:
                inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'

        elif isinstance(stimulus, stim.EncastreBC):
            inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
            node_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]               
            for ent in node_labels:
                inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'

        elif isinstance(stimulus, stim.RollerSupport):
            inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
            node_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]               
            for ent in node_labels:
                inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'

        elif isinstance(stimulus, stim.PinnedSupport):
            inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
            node_labels = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]               
            for ent in node_labels:
                inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'                                   
        
        file.write(inp_deck)


def _write_element_output(file, set_name="ALLELEMENTS", frd=False):
    """Write part of inp deck that dictates what is written in the dat file by Calculix ccx for the
    elements.

    - This current configuration will mean the element stresses.

    Args:
        file: Inp file.

    Returns:
        None
    """
    inp_deck = f"*EL PRINT, ELSET={set_name}\nS, ENER, EVOL, E, ME\n"
    #inp_deck = f"*EL PRINT, ELSET={set_name}\nS\n"
    if frd:
        inp_deck += f"*EL FILE\nS\n"
    file.write(inp_deck)


def _write_all_node_output(file, output=None):
    """Write part of inp deck that dictates what is written in the dat file by Calculix ccx for the
    nodes.

    - This current configuration will mean the node displacements.

    Args:
        file: Inp file.

    Returns:
        None
    """
    inp_deck = '*NODE FILE, OUTPUT=3D\n'
    if output == 'temp':
        inp_deck += 'NT\n'
    elif output == 'disp':
        inp_deck += 'U\n'
    else:
        inp_deck += 'U, NT\n'
    inp_deck += f'*NODE PRINT, NSET=ALLNODES\n'
    inp_deck += f'RF, NT, U\n'  
    #inp_deck += f'U\n'        

    file.write(inp_deck)