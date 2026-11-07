"""Module contains functions related to completing an inp deck outputted from GMSH
so it can be used by Calculix ccx for thermal analysis"""
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
        elif isinstance(stimulus, stim.TemperaturePlanar):
            stimulus_script += f"*NSET, NSET={stimulus.group_name}\n"
            if isinstance(entity[0], Edge):
                nodes = [ent.start_point.id+1, ent.end_point.id+1]
                stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{nodes[0]},{nodes[1]}\n"            
            else:
                if len(entity) > 16:
                    lines = [entity[i:i+16] for i in range(0, len(entity), 16)]
                    for ent in lines:
                        ent= np.asarray(ent)
                        stimulus_script += np.array2string(ent, separator=', ').strip("[]") +'\n'
                else:
                    for ent in entity:
                        stimulus_script += f"{ent},"
                    stimulus_script += "\n"                

        elif isinstance(stimulus, stim.TemperaturePoint):
            for ent in entity:
                if isinstance(ent, Point):
                    stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{ent.id + 1}\n"
        elif isinstance(stimulus, stim.ConcentratedHeatFlux):
            for ent in entity:
                stimulus_script += f"*NSET, NSET={stimulus.group_name}\n{ent}\n"

    file.write(stimulus_script)


def _write_generate_node_set(file, node_labels):
    """Write node set for all points in inp.

    Args:
        file: Inp file.
        node_labels: Node labels from GMSH.
    
    Returns:
        None
    """
    inp_deck = (
        f'*NSET, NSET=ALLNODES, GENERATE\n'
        f'{min(node_labels)}, {max(node_labels)}, 1\n'
    )
    file.write(inp_deck)


def _write_generate_element_set(file, element_labels):
    """Write element set for all elements in inp.

    Args:
        file: Inp file.
        element_labels: Element labels from GMSH.
    
    Returns:
        None
    """
    inp_deck = (
        f'*ELSET, ELSET=ALLELEMENTS, GENERATE\n'
        f'{min(element_labels)}, {max(element_labels)}, 1\n'
    )
    file.write(inp_deck)


def _write_beam_sections(file, element_labels, organism):
    """Write beam section for each edge in inp.

    Args:
        file: Inp file.
        element_labels: Element labels from organism beam mesh property.
        organism: Organism object.
    
    Returns:
        Dictionaries of materials in organism. Key is material name and value is material object.
    """
    materials = {}
    edges = organism.edges
    edge_el_ids = {}
    for edge in edges.values():
        edge_el_ids[edge._el_id] = edge.id

    inp_deck = ''

    for label in element_labels:

        edge_id = edge_el_ids[label]

        material = edges[edge_id].material
        cs_area = edges[edge_id].cs_area

        solid_section = (
            f'*BEAM SECTION, ELSET=L{label}, MATERIAL="{material.name}", SECTION=CIRC\n'
            f'{2 * np.sqrt(cs_area/np.pi)}\n'
        )
        inp_deck += solid_section

        if material.name not in materials:
            materials[material.name] = material

    file.write(inp_deck)

    return materials


def _write_plate_shell_section(file, organism):

    materials = {}
    faces = organism.faces.values()

    inp_deck = ''
    for face in faces:
        solid_section = (
            f'*SHELL SECTION, ELSET=Surface{face.id}, MATERIAL="{face.material.name}"\n'
            f'{round(face.thickness, 2)}\n'
        )
        inp_deck += solid_section

        if face.material.name not in materials:
            materials[face.material.name] =face.material

    file.write(inp_deck)

    return materials


def _write_physical_constants(file):
    """Physical constants required for certain thermal analyses
    Absolute Zero expressed in K
    Stefan boltzmann constant expressed in W / (mm^2 K^4)
    """
    inp_deck = '*PHYSICAL CONSTANTS,ABSOLUTE ZERO=-273.15,STEFAN BOLTZMANN=5.669E-2\n'

    file.write(inp_deck)


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

    Args:
        file: Inp file.
        stimuli: List of environmental stimuli
    
    Returns:
        None
    """
    inp_deck = ''

    for stimulus in stimuli_entities.keys():
        if isinstance(stimulus, stim.InitialTemperature):
            inp_deck += stimulus.get_inp_deck()

    file.write(inp_deck)


def _write_step(file, stimuli_entities):
    """Write step section of inp.
    
    - This current configuration will mean a static step is used.

    Args:
        file: Inp file.
        stimuli entities: dictionary of stimuli keys and geometric entities (values)
    
    Returns:
        None
    """
    thermal_stim = [stimulus for stimulus in stimuli_entities.keys() if
               isinstance(stimulus, stim.TemperaturePlanar) or
               isinstance(stimulus, stim.TemperaturePoint) or
               isinstance(stimulus, stim.DistributedHeatFlux) or
               isinstance(stimulus, stim.ConcentratedHeatFlux)]
    if len(thermal_stim) >=1:
        #Run a heat transfer as the analysis step
        inp_deck = (
            '*STEP\n'
            '*HEAT TRANSFER, STEADY STATE\n'
            '0.1, 1.0\n'
        )
    else:
        #Run a static as structural only
        inp_deck = (
            '*STEP\n'
            '*STATIC\n'
            '0.1, 1.0\n'
        )
    file.write(inp_deck)


def _write_radiate(file, materials, elements):
    """Write emissivity properties of elements to inp.

    Args:
        file: inp file
        materials: materials of cells for emissivity value
        elements: 'all elements' or element set name

    Returns:
        None
    """

    inp_deck = '*RADIATE\n'
    for _ , material in materials.items():
        emissivity = material.emissivity

    if elements == 'all':
        inp_deck += f'ALLELEMENTS, R1, 293.0, {emissivity}\n'
    else:
        inp_deck += f'{elements}, R1, 293.0, {emissivity}\n'

    file.write(inp_deck)

def _write_stimuli(file, stimuli_entities, dim):
    """Write stimuli into inp.

    Args:
        file: Inp file.
        stimuli_points: Dictionary associating environmental stimuli with points in organism.
        Key is Point object, value is list of associated stimuli.
    
    Returns:
        None
    """
    inp_deck = ''

    for stimulus in stimuli_entities.keys():     
        if isinstance(stimulus, stim.TemperaturePlanar) or\
            isinstance(stimulus, stim.TemperaturePoint):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.Temperature):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.FixedSupportPlanar):
            if dim == 1:
                inp_deck += stimulus.get_inp_deck('edge')
            elif dim == 2:
                inp_deck += stimulus.get_inp_deck('face')
        elif isinstance(stimulus, stim.FilmCondition):
            if len(stimuli_entities[stimulus]) > 1:
                inp_deck += stimulus.get_inp_deck(True)
            else:
                inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.DistributedHeatFlux):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.ConcentratedHeatFlux):
            inp_deck += stimulus.get_inp_deck()
        elif isinstance(stimulus, stim.FilmConditionEdge):
            inp_deck += stimulus.get_inp_deck()

    file.write(inp_deck)


def _write_all_node_output(file):
    """Write part of inp deck that dictates what is written in the dat file by Calculix ccx for
    the nodes.
    
    - This current configuration will mean the node temperatures.

    Args:
        file: Inp file.
    
    Returns:
        None
    """
    inp_deck = (
        '*NODE PRINT, NSET=ALLNODES\n'
        'NT, U\n'
    )
    file.write(inp_deck)


# def _print_all_node_output(file):
#     """Write part of inp deck that dictates what is written in the dat file by Calculix ccx for
#       the nodes.

#     - This current configuration will mean the node displacements.

#     Args:
#         file: Inp file.

#     Returns:
#         None
#     """
#     inp_deck = (
#         f'*NODE FILE, NSET=ALLNODES\n'
#         f'NT\n'
#     )
#     file.write(inp_deck)


def _write_element_output(file, set_name=None):
    """Write part of inp deck that dictates what is written in the dat file by Calculix ccx for the
    elements.
    
    - This current configuration will mean the element stresses.

    Args:
        file: Inp file.
    
    Returns:
        None
    """
    if set_name is None:
        inp_deck = (
            '*EL PRINT, ELSET=ALLELEMENTS\n'
            # 'HFL\n'
            'HFL, S, EVOL, ENER\n'
        )
    else:
        inp_deck = f"*EL PRINT, ELSET={set_name}\nHFL\n"
              
    file.write(inp_deck)

def _write_stimulus_groups(file, stimulus_entities):
    """
    Write the physical groups (node or element set) from the stimulus
    entities and organism data by matching stimulus with node/element
    id in organism
    """
    
    for stimulus, entity_list in stimulus_entities.items():
        for entity in entity_list:
            if isinstance(entity, Point):
                inp_deck = f'*NSET, NSET={stimulus.group_name}\n{entity.id + 1}\n'
                file.write(inp_deck)
            elif entity == "eplate":
                inp_deck = f'*ELSET, ELSET={stimulus.group_name}\neplate\n'
                file.write(inp_deck)
            elif isinstance(stimulus, stim.FilmConditionEdge):
                inp_deck = f'*ELSET, ELSET={stimulus.group_name}\n{entity}\n'
                file.write(inp_deck)
    
        if len(entity_list) > 16:
            if isinstance(stimulus, stim.TemperaturePlanar):
                inp_deck = f'*NSET, NSET={stimulus.group_name}\n'
                node_lines = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]
                for ent in node_lines:
                    inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'
            if isinstance(stimulus, stim.DistributedHeatFlux):
                inp_deck = f'*ELSET, ELSET={stimulus.group_name}\n'
                ele_lines = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]
                for ent in ele_lines:
                    inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'
            if isinstance(stimulus, stim.FixedSupportPlanar):
                inp_deck = f'NSET, NSET={stimulus.group_name}\n'
                node_lines = [np.array(entity_list[i:i + 5]) for i in range(0, len(list(entity_list)), 5)]
                for ent in node_lines:
                    inp_deck += np.array2string(ent, separator=', ').strip("[]") +'\n'                               
            else:
                pass
                # inp_deck = f'*ELSET, ELSET={stimulus.group_name}\n' + str(entity_list).strip("[]") +'\n'
        
        file.write(inp_deck)
