"""
This module contains the functions for interfacing with the Calculix solver ccx, which can be used
to act upon the organism
- ccx needs to be pre-installed on system to use this module.
- On Linux Debian based distros install using $ sudo apt-get install calculix-ccx
- Else, Calculix can be installed from source from: https://github.com/calculix/cae
"""
import copy
from collections import defaultdict, OrderedDict
import os
import subprocess
import numpy as np
from src.utils.errors import CcxElemNumError, InvalidDesignError, CcxElemSizeError, CcxCutBackError, CcxElemJacobianError

def run_ccx(file_path: str, silent: bool=True):
    """Run Calculix solver ccx on .inp file.

    - Runs the inp file in ccx and removes the redundant files that are produced.

    Args:
        file_path: File path of .inp file.
        silent: If silent is True the output from ccx will not appear in terminal, it is written to
        log file.

    Returns:
        None

    """
    if file_path.endswith(".inp"):
        file_path = file_path[:-len(".inp")]

    if silent:
        with open(f"{file_path}.log", "w") as log_file:
            subprocess.run(["ccx", file_path], stdout=log_file)

    else:
        subprocess.run(["ccx", file_path])

    if os.path.exists(f"{file_path}.12d"):
        os.remove(f"{file_path}.12d")

    if os.path.exists(f"{file_path}.cvg"):
        os.remove(f"{file_path}.cvg")

    # if os.path.exists(f"{file_path}.frd"):
    #     os.remove(f"{file_path}.frd")

    if os.path.exists(f"{file_path}.sta"):
        os.remove(f"{file_path}.sta")


def _load_dat_file(file_path):
    """Read a .dat file and load into memory.

    Args:
        file_path: File path of .dat file.

    Returns:
        Lines of .dat file.

    """
    if not file_path.endswith(".dat"):
        dat_file_path = f"{file_path}.dat"

        if os.stat(dat_file_path).st_size != 0:
            with open(dat_file_path) as dat_file:
                lines = [line.strip() for line in dat_file if not line.isspace()]
        else:
            log_file_path = f"{file_path}.log"
            with open(log_file_path) as dat_file:
                lines = [line for line in dat_file]

                if len(lines) == 0:
                    raise InvalidDesignError

                for line in lines:
                    if line.strip().split(" ")[0] == "*ERROR":
                        if line == " *ERROR in gen3dnor: more than 100\n":
                            raise CcxElemNumError
                        if line == " *ERROR in gen3dnor: size of estimated\n":                      
                            raise CcxElemSizeError
                        elif line == " *ERROR reading *SOLID SECTION: normal\n":
                            raise InvalidDesignError
                        elif line == "*ERROR: too many cutbacks\n":
                            raise CcxCutBackError
                        elif line == " *ERROR in e_c3d_th: nonpositive jacobian\n":
                            raise CcxElemJacobianError
                        else:
                            print(line)

                        break
            lines = None

    return lines


def get_tensor_matrix(voigt_mat):
    """Gets a tensor matrix from the voight matrix notation.

    - Transform the voight matrix notation from the dat file to a tensor matrix.
    - The tensor matrix is symmetrical about the diagonal, whereas the voight is a
      shorthand notation of this.

    Args:
        voigt_mat: Voight matrix notation from the dat file.

    Returns:
        Tensor matrix.
    """
    tensor_mat = np.identity(3)
    np.fill_diagonal(tensor_mat, tensor_mat.diagonal() * voigt_mat[:3])
    tensor_mat[0, 1] = voigt_mat[3]
    tensor_mat[1, 0] = voigt_mat[3]
    tensor_mat[0, 2] = voigt_mat[4]
    tensor_mat[2, 0] = voigt_mat[4]
    tensor_mat[1, 2] = voigt_mat[5]
    tensor_mat[2, 1] = voigt_mat[5]

    return tensor_mat


def _parse_dat_lines(lines, section_title, set_name, time_stamp=None):
    """Parses the lines read from the .dat file for specific section title.

    - Parses the lines of the .dat file for the given section title.
    - If the section title is found, the lines relating to the section title are extracted &
    removes from the
     lines list.

    Args:
        lines: Lines of .dat file.
        section_title: Section title of lines to extract.

    Returns:
        Lines of .dat file with given section removed and lines of .dat file relating to given
        section title.
    """
    section_lines = []
    
    if time_stamp is None:
        this_time_stamp = 0.0
    
    for i, line in enumerate(lines):
        if line.split("(")[0].strip() == section_title and line.split("set ")[1].split(" ")[0].strip() == set_name.upper() \
            or line.split(" ")[0].strip() == section_title and line.split("set ")[1].split(" ")[0].strip() == set_name.upper():
            if time_stamp is None:
                next_time_stamp = float(line.split("time ")[-1].strip())
                if next_time_stamp > this_time_stamp:
                    this_time_stamp = next_time_stamp
                    start_line = i + 1
                    section_lines = []
                    for j, nxt_line in enumerate(lines[start_line:]):
                        if nxt_line.split("(")[0].split(" ")[0].isalpha():
                            break
                        elif i + j + 1 == len(lines):
                            break
                        else:
                            section_lines.append(nxt_line)
            else:
                if line.split("time ")[-1].strip() == time_stamp:
                    start_line = i + 1
                    section_lines = []
                    for j, nxt_line in enumerate(lines[start_line:]):
                        if nxt_line.split("(")[0].replace(" ", "").isalpha():
                            break
                        elif i + j + 1 == len(lines):
                            break
                        else:
                            section_lines.append(nxt_line)
                    break      
    return lines, section_lines


def _get_critical_buckling_factor(lines):
    """Read dat file from buckling analysis for critical buckling factor (at mode 1), 
    
    Attributes:
        lines: lines read from dat file

    Returns:
        critical buckling factor
    """

    for line in lines:
        if line.split(" ")[0] == '1':
            critical_buckling_factor = line.split(" ")[-1]

    return critical_buckling_factor


def _get_face_element_values(section_lines, out_array):
    """Gets element values from the given section lines and averages for elements.

    - Loops over given section lines & extracts the required data.
    - From the data, the element id, integration point number & components of output values are 
    extracted.
    - The integration point values are summated for a given element in the output array.
    - The this_element_id variable keeps track which element to index in the output array.
    - If the this_element_id is different than the element_id, the this_element_id is updated
    - And the row in the output array relating to the previous element is averaged using the number 
    of integration points

    Args:
        section_lines: Lines of .dat file relating to given section title.
        out_array: Array to which the averaged integration point values for each element will be assigned.

    Returns:
        Updated output array with averaged integration point values for each element.
    """

    this_element_id = float(section_lines[0].split(" ")[0])
    ind = 0

    for line in section_lines:
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]

        element_id = int(data[0])

        if element_id > this_element_id:
            out_array[ind] /= int_point
            this_element_id = element_id
            ind += 1

        int_point = int(data[1])
        try:
            int_pt_data = [float(i) for i in data[2:]]
        except ValueError:
            int_pt_data = [float(i) for i in data[2:-1]]
        out_array[ind] += int_pt_data

    out_array[ind] /= int_point

    return out_array


def _get_element_volumes(section_lines):
    """Gets the element volumes from the inp file

    Args:
        section_lines: Lines of .dat file relating to given section title.

    Returns:
        element_vol_dict: Ordered dict with element id as key and volume as value
    """
    element_vol_dict = OrderedDict()

    for ind, line in enumerate(section_lines):
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]

        el_id = int(data[0])
        element_vol = float(data[1])
        element_vol_dict[el_id] = element_vol

    return element_vol_dict


def _get_inp_point_ids(file_path):
    """Gets a dictionary linking the inp node id to the organism point id.

    Args:
        file_path: Inp file path.

    Returns:
        A dictionary where the keys are the inp node ids and the values are the organism point ids.
    """
    point_dict = {}
    point_count = 0
    node_section = False

    with open(f"{file_path}.inp") as inp_file:
        for _ , line in enumerate(inp_file):
            if line == "*NODE\n":
                node_section = True
                continue
            if node_section:
                if line == "******* E L E M E N T S *************\n":
                    break
                else:
                    data = line.split(", ")
                    point_dict[int(data[0])] = point_count
                    point_count += 1

    return point_dict


def _get_point_displacements(lines, file_path, num_points, tolerance=1e-9):
    """Get displacements for each point.

    - Parses the lines from the .dat file for data relating to the 'displacements'.
    - Get the inp node and point id dictionary.
    - Creates an array to hold the displacements for each point.
    - Loops over given section lines & assigns displacement to given point.
    - Where displacements are below tolerance assign 0.0 to displacements.

    Args:
        lines: Lines of .dat file.
        organism: Organism object.
        file_path: Inp file path.
        tolerance: Tolerance value.

    Returns:
        Displacements for each point in organism.
    """
    section_title = "displacements"
    lines, section_lines = _parse_dat_lines(lines, section_title, 'ALLNODES')
    point_id_dict = _get_inp_point_ids(file_path)

    point_displacements = np.zeros((num_points, 3))

    for line in section_lines:
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]

        mesh_point_id = int(data[0])
        point_data = [float(i) for i in data[1:]]
        try:
            point_id = point_id_dict[mesh_point_id]
            point_displacements[point_id] = point_data
        except KeyError or IndexError:
            # with fin addition point id and node id don't match (key error)
            # for beam mesh, midpoints should be ignored
            pass
        

    point_displacements = np.where(np.abs(point_displacements) < tolerance, 0.0,point_displacements)

    return point_displacements


def _get_mem_buckling(mem_stresses, mem_lengths, mem_areas, mem_mods, moment_areas):
    """Calculate if the edge will buckle given its stress.

    - Compares critical stress with the stresses of the edge.
    - Only check for buckling on edges with compressive loads.
    - Calculate the percentage of the compressive load with the critical stress.
    - If the compressive load is above the critical stress set the percentage to 100%.
    - If the load is tensile, then the percentage will be 0%.

    Args:
        mem_stresses: Array of stresses for each edge.
        mem_lengths: Array of lengths for each edge.
        mem_areas: Array of cross-sectional area for each edge.
        mem_mods: Array of Young's modulus for each edge.
        moment_areas: Array of second moment of areas for each edge.
        fitness: if True return ratio of stress to critical stress. If false, i.e. a constraint, 
        return inverse

    Returns:
        Percentages for each edge of how close it is to buckling.
    """

    np.seterr(invalid='ignore')
    mem_crit_stresses = np.pi ** 2 * mem_mods * moment_areas / np.power(mem_lengths, 2) / mem_areas
    mem_stresses = np.where(mem_stresses > 0, 0, mem_stresses)
    buckling = np.abs(mem_stresses) / mem_crit_stresses

    return np.where(buckling > 1.0, 1.0, buckling)


def _get_face_strain_energies(lines, faces):
    """
    Get the total strain enegry values of the faces of the organism ( from their element values)

    Args:
        lines: .dat file lines
        faces: number of faces / elements in organism / analysis

    Returns:
        face_strain_energy: array of face strain energy values, where rows are indexed faces and
        columns are the strain energy values calculated for the face
    """

    face_strain_energy = np.zeros((faces))

    # get se density values for the elements of each face
    sed_section_title = "internal energy density"
    set_name = "ALLELEMENTS"
    #time_stamp = "0.1000000E+01"
    lines, sed_section_lines = _parse_dat_lines(lines, sed_section_title, set_name)
    element_sed_values = _get_face_element_values(sed_section_lines,
                                                  copy.deepcopy(face_strain_energy))

    # get volume of each element of each face
    vol_section_title = "volume"
    set_name = "ALLELEMENTS"
    lines, vol_section_lines = _parse_dat_lines(lines, vol_section_title, set_name)
    element_vol_dict = _get_element_volumes(vol_section_lines)
    element_vol_values = np.asarray(list(element_vol_dict.values()))

    # calculate strain energy values from element se density and volume
    element_se_values = element_sed_values / element_vol_values

    return element_se_values


def _get_face_stress(lines, faces, set_name=None):
    """
    Get the stress values of the faces of the organism (from their element values)

    Args:
        lines: .dat file lines

    Returns:
        face_stresses: m by n array of face stress values, where rows are indexed faces and columns 
        are principal stress values of the face
    """

    face_stress = np.zeros((faces))

    out_array = np.zeros((faces, 6))

    # get stress values for the element
    section_title = "stresses"
    if set_name == None:
        set_name = "ALLELEMENTS"
    lines, section_lines = _parse_dat_lines(lines, section_title, set_name)
    element_stress_values = _get_face_element_values(section_lines, copy.deepcopy(out_array))


    for ind, stress_comps in enumerate(element_stress_values):
        tensor_mat = get_tensor_matrix(stress_comps)
        principal_stresses = np.linalg.eigvals(tensor_mat)
        principal_stress = principal_stresses[np.abs(principal_stresses).argmax()]
        
        #xx = principal_stresses[0]
        #yy = principal_stresses[1]       
        #zz = principal_stresses[2]
        #principal_stress = np.sqrt(((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2)/2)      
        
        face_stress[ind] = principal_stress
        #face_stress[ind] = principal_stresses[1]

    return face_stress


def _get_beam_stress(organism, lines, set_name=None) -> np.ndarray:
    """Gets element values from the given section lines and averages for elements.

    - Loops over given section lines & extracts the required data.
    - From the data, the element id, integration point number & components of output values are
    extracted.
    """

    section_title = "stresses"
    if set_name is None:
        set_name = "ALLELEMENTS"
    lines, section_lines = _parse_dat_lines(lines, section_title, set_name)

    compression = np.zeros((len(organism.edges),9,6))
    neutral = np.zeros((len(organism.edges),9,6))
    tension = np.zeros((len(organism.edges),9,6))

    stresses = defaultdict(list)
    for j, line in enumerate(section_lines):
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]
                
        # change el id to edge id (member id)
        el_id = int(data[0])

        stresses[el_id].append([float(i) for i in data[2:]])

    """ Sxx as marker"""
    for el_id, stress_comp in stresses.items():

        non_virtual_edges = [edge for edge_id, edge in sorted(organism.edges.items())
                             if not edge.virtual]
        for ind, edge in enumerate(non_virtual_edges):
            if edge._el_id == el_id:
                member_id = edge.id
                break        
        k=0
        for k in range(10):

            if k < 3:
                compression[member_id][k] = stresses[el_id][k]
                neutral[member_id][k] = stresses[el_id][k+3]
                tension[member_id][k] = stresses[el_id][k+6]

            elif 3 <= k < 6:
                compression[member_id][k] = stresses[el_id][k+6]
                neutral[member_id][k] = stresses[el_id][k+9]
                tension[member_id][k] = stresses[el_id][k+12]

            elif 6 <= k < 9:
                compression[member_id][k] = stresses[el_id][k+12]
                neutral[member_id][k] = stresses[el_id][k+15]
                tension[member_id][k] = stresses[el_id][k+18]

    # """ Syy as marker"""
    # k=0
    # for k in range(len(stresses)+1):

    #     if k < 3:
    #         tension[k] = stresses[k]
    #         neutral[k] = stresses[k+3]
    #         compression[k] = stresses[k+6]

    #     elif 3 <= k < 6:
    #         compression[k] = stresses[k+6]
    #         neutral[k] = stresses[k+9]
    #         tension[k] = stresses[k+12]

    #     elif 6 <= k < 9:
    #         tension[k] = stresses[k+12]
    #         neutral[k] = stresses[k+15]
    #         compression[k] = stresses[k+18]    

    compression = np.mean(compression, 1)
    neutral = np.mean(neutral, 1)
    tension = np.mean(tension, 1)

    abs_beam_stresses = np.zeros(len(organism.edges))
    beam_stresses = np.zeros(len(organism.edges))
    for edge, element_voight in enumerate(compression):
        tensor_mat = get_tensor_matrix(element_voight)
        principal_stresses = np.linalg.eigvals(tensor_mat)
        #print(principal_stresses[np.abs(principal_stresses).argmax()])
        abs_beam_stresses[edge] += abs(principal_stresses[np.abs(principal_stresses).argmax()])
        beam_stresses[edge] += principal_stresses[np.abs(principal_stresses).argmax()]

    for edge, element_voight in enumerate(neutral):
        tensor_mat = get_tensor_matrix(element_voight)
        principal_stresses = np.linalg.eigvals(tensor_mat)
        #print(principal_stresses[np.abs(principal_stresses).argmax()])
        abs_beam_stresses[edge] += abs(principal_stresses[np.abs(principal_stresses).argmax()])
        beam_stresses[edge] += principal_stresses[np.abs(principal_stresses).argmax()]

    for edge, element_voight in enumerate(tension):
        tensor_mat = get_tensor_matrix(element_voight)
        principal_stresses = np.linalg.eigvals(tensor_mat)
        #print(principal_stresses[np.abs(principal_stresses).argmax()])
        abs_beam_stresses[edge] += abs(principal_stresses[np.abs(principal_stresses).argmax()])
        beam_stresses[edge] += principal_stresses[np.abs(principal_stresses).argmax()]

    # Account for the constant 56% discrepency in CCX beam stress cal here for accurate comparison
    # between stress (corrected here) and critial buckling stress (analytical)
    abs_beam_stresses = abs_beam_stresses * 1.56
    beam_stresses = beam_stresses * 1.56

    return abs_beam_stresses, beam_stresses


def get_total_volume(mem_lengths, mem_areas):
    """Get the total volume of the organism.

    Args:
        mem_lengths: Array of lengths for each edge.
        mem_areas: Array of cross-sectional areas for each edge.

    Returns:
        A single total volume float.
    """
    return np.sum(mem_lengths * mem_areas)


def get_total_volume_dim_two(organism=None):
    """Get the total volume of the organism.

    Args:
        mem_lengths: Array of lengths for each edge.
        mem_areas: Array of cross-sectional areas for each edge.

    Returns:
        A single total volume float.
    """
    face_areas = []
    thicknesses = []
    faces = organism.faces
    for _ , face in sorted(faces.items()):
        if not face.virtual:
            face_areas.append(face.surface_area)
            thicknesses.append(face.thickness)
    face_areas = np.asarray(face_areas)
    thicknesses = np.asarray(thicknesses)
    return np.sum(face_areas * thicknesses)


def get_total_strain_energy(mem_strain_energies):
    """Get the total strain energy of the organism.

    Args:
        mem_strain_energies: Array of strain energy values for each edge.

    Returns:
        A single total strain energy float.
    """
    return np.sum(mem_strain_energies)


def get_max_deflection(point_displacements, direction=None):
    """Get the max deflection of the organism.

    - To calculate the max deflection, the maximum displacement value in any axis for all the points is found.

    Args:
        point_displacements: Array of the x, y and z displacements for each point in the organism.

    Returns:
        The max deflection as a float.
    """
    if direction == "x":
        max_deflection = np.max(np.abs(point_displacements)[0])
    elif direction == "y":
        max_deflection = np.max(np.abs(point_displacements)[1])
    elif direction == "z":
        max_deflection = np.max(np.abs(point_displacements)[2])
    else:
        max_deflection = np.max(np.abs(point_displacements))

    return max_deflection


def get_total_deflection(point_displacements):
    """Get the total deflection of the organism.

    - To calculate the total deflection, the sum of each displacement value in any axis for all the points is found.

    Args:
        point_displacements: Array of the x, y and z displacements for each point in the organism.

    Returns:
        The max deflection as a float.
    """
    return np.sum(np.abs(point_displacements))


def get_max_stress(mem_stresses):
    """Get the max stress of the organism.

    Args:
        mem_stresses: Array of the stresses for each member/edge in the organism.

    Returns:
        The max stress as a float.
    """
    if isinstance(mem_stresses, np.ndarray):
        return np.max(np.abs(mem_stresses))


def get_average_stress(mem_stresses):
    """Get the average stress of the organism.

    Args:
        mem_stresses: Array of the stresses for each member/edge in the organism.

    Returns:
        The average stress as a float.
    """
    if isinstance(mem_stresses, np.ndarray):
        return np.average(np.abs(mem_stresses))


def get_total_stress(mem_stresses):
    """Get the total stress of the organism.

    Args:
        mem_stresses: Array of the stresses for each member/edge in the organism.

    Returns:
        The total stress as a float.
    """
    if isinstance(mem_stresses, np.ndarray):
        return np.sum(np.abs(mem_stresses))

def _get_beam_displacements(lines, organism):
    """
    Get the displacement values of the edges of the organism

    Args:
        lines: .dat file lines
        file_path: .inp file path

    Returns:
        beam_displacements: dict with beam id as key and list of node displacements of that edge
    """

    section_title = "displacements"
    lines, section_lines = _parse_dat_lines(lines, section_title, 'ALLNODES')

    beam_displacements = np.zeros((len(organism._beam_nodes), 3))

    for line in section_lines:
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]

        mesh_node_id = int(data[0])
        node_disp = np.array([float(i) for i in data[1:]])

        for beam_id, nodes in organism._beam_nodes.items():
            if mesh_node_id in nodes:
                line_id = beam_id - len(organism.edges) - len(organism.points) - 1
                beam_displacements[line_id] += node_disp

    # average the beam displacement from 3 node displacements
    beam_displacements = beam_displacements / 3

    return np.amax(beam_displacements, axis=1)

def _get_shell_displacements(lines, organism):
    """
    Get the displacement values of the faces of the organism

    Args:
        lines: .dat file lines
        file_path: .inp file path

    Returns:
        shell_displacements: dict with face id as key and list of node displacements of that edge
    """

    section_title = "displacements"
    lines, section_lines = _parse_dat_lines(lines, section_title, 'ALLNODES')

    shell_displacements = np.zeros((len(organism.cells), 3))

    for line in section_lines:
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]

        mesh_node_id = int(data[0])
        node_disp = np.array([float(i) for i in data[1:]])

        for shell_id, nodes in organism._face_nodes.items():
            if mesh_node_id in nodes:
                keys = [k for k, v in organism.face_shell_data.items() if v == shell_id]
                shell_displacements[keys[0]] += node_disp

    # average the shell displacement from 3 node displacements
    shell_displacements = shell_displacements / 3
    
    return np.amax(shell_displacements, axis=1)

def _get_all_displacements(lines, organism):
    section_title = "displacements"
    lines, section_lines = _parse_dat_lines(lines, section_title, 'ALLNODES')

    node_displacements = np.zeros((len(organism.edges) + len(organism.points), 3))

    for line in section_lines:
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]

        mesh_node_id = int(data[0]) - 1
        node_disp = np.array([float(i) for i in data[1:]])

        node_displacements[mesh_node_id] = node_disp

    return node_displacements


if __name__ == "__main__":

    """
    The code below will load a specific dat file from ried_foss folder and obtain
    state data using the functions in this file.
    The example shows face_stresses being read.
    """
    file_path = os.path.join(os.getcwd(), 'dat file name here e.g. 59-59-5')
    lines = _load_dat_file(file_path)
    faces = {}
    face_stresses = _get_face_stress(lines, 178, "eplate")

    print("End")
