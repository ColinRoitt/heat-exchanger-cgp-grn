"""
This module contains the functions for interfacing with the Calculix solver ccx, which can be used
to act upon the organism.

- ccx needs to be pre-installed on system to use this module.
- On Linux Debian based distros install using $ sudo apt-get install calculix-ccx
- Else, Calculix can be installed from source from: https://github.com/calculix/cae
"""

import os
import subprocess
import csv
from collections import OrderedDict
import numpy as np
from dataclasses import astuple
from src.utils.errors import CcxElemNumError, InvalidDesignError
import src.environment.stimulus as stim


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

    if os.path.exists(f"{file_path}.frd"):
        os.remove(f"{file_path}.frd")

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
                for line in dat_file:
                    if line.strip().split(" ")[0] == "*ERROR":
                        if line == " *ERROR in gen3dnor: more than 100\n":
                            raise CcxElemNumError
                        if line == " *ERROR in e_c3d_th: nonpositive jacobian\n":
                            raise InvalidDesignError
                        elif line == " *ERROR reading *SOLID SECTION: normal\n":
                            raise InvalidDesignError
                        elif line == " *ERROR in gen3dnor: size of estimated\n":
                            raise InvalidDesignError
                        else:
                            print(line)

                        break
            lines = None

    return lines


def _parse_dat_lines(lines, section_title, set_name, time_stamp=None):
    """Parses the lines read from the .dat file for specific section title.

    - Parses the lines of the .dat file for the given section title.
    - If the section title is found, the lines relating to the section title are extracted & removes
    from the lines list.

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
                        if nxt_line.split("(")[0].replace(" ", "").isalpha():
                            break
                        elif i + j + 1 == len(lines):
                            break
                        else:
                            section_lines.append(nxt_line)
            # else:
            #     if line.split("time ")[-1].strip() == time_stamp:
            #         start_line = i + 1
            #         section_lines = []
            #         for j, nxt_line in enumerate(lines[start_line:]):
            #             if nxt_line.split("(")[0].replace(" ", "").isalpha():
            #                 break
            #             elif i + j + 1 == len(lines):
            #                 break
            #             else:
            #                 section_lines.append(nxt_line)
            #         break      
    return lines, section_lines



def _get_heat_flux(lines, num_faces, set_name=None):
    """Get the heat flux of the organism in the x, y and z directions
    
    - heat flux calculated at the end of the step (assume t=1.0).
    - The average of the integ.pts of all the elements, is averaged for x, y and z directions.

    Args:
        lines: .dat file lines 
        num_faces: number of faces/elements in organism/analysis
        
    Returns:
        list of heat flux values in z (assumed perpendicular to surface)
    """
    face_heat_flux = np.zeros((num_faces))
    out_array = np.zeros((num_faces, 3))


    hfl_section_title = "heat flux"
    if set_name is None:
        set_name = "ALLELEMENTS"
    #time_stamp = "0.1000000E+01"
    lines, hfl_section_lines = _parse_dat_lines(lines, hfl_section_title, set_name)
    if len(hfl_section_lines) > 0:
        # Only process hfl if there are values in the dat file
        element_hfl_values = _get_face_element_values(hfl_section_lines, out_array)
        for ind, hfl in enumerate(element_hfl_values):
            heat_flux = np.linalg.norm(hfl)
            face_heat_flux[ind] = heat_flux

    return face_heat_flux


def _get_node_temperatures(lines, file_path):
    """Get the node temperatures of the organism at the end of the analysis
    
    - nodal temps read from the end of the step (assume t=1.0)

    Args:
        lines: .dat file lines 
        
    Returns:
        creates output file of node temperatures
    """

    node_temps = {}
    output_file = f'{file_path}.csv'
    nt_section_title = "temperatures"
    set_name = "ALLNODES"
    #time_stamp = "0.1000000E+01"
    lines, nt_section_lines = _parse_dat_lines(lines, nt_section_title, set_name, time_stamp=None)
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for line in nt_section_lines:
            node_tag = int(line.split(" ")[0].strip())
            node_temp = float(line.split(" ")[-1].strip())
            node_temps[node_tag] = node_temp
            writer.writerow([node_tag, node_temp])

    return node_temps 


def get_total_heat_flux(heat_flux_results):
    """Get the total heat flux of the organism.
    
    Args: 
        heat_flux_results: list of face heat flux values in x, y and z
    """

    try:
        total_heat_flux = 1 / np.sum(heat_flux_results).tolist()
    except ZeroDivisionError:
        print("Total heat flux can not be calculated from this environment")
        total_heat_flux = 0
        
    return total_heat_flux

def get_average_heat_flux(heat_flux_results):
    """Get the average heat flux of the organism.
    
    Args: 
        heat_flux_results: list of face heat flux values in x, y and z
    """

    try:
        ave_heat_flux = 1 / np.mean(heat_flux_results).tolist()
    except ZeroDivisionError:
        print("Average heat flux can not be calculated from this environment")
        ave_heat_flux = 0
        
    return ave_heat_flux


def get_total_heat_transfer(heat_flux_results, organism, stimuli):
    """Get the toal heat transfer through the organism
    
    Args:
        heat_flux_results: list of face heat flux values in x, y, z
        organism: instance of organism
        stimuli: list of environment stimuli
    """
    # Get temperature stimulus (largest) and get it's normal
    all_temp_stim = [this_stim for this_stim in stimuli
                     if isinstance(this_stim, stim.TemperaturePlanar)]
    this_stim_mag = 0
    for temp in all_temp_stim:
        if temp.magnitude > this_stim_mag:
            this_stim_mag = temp.magnitude
            temp_stim = temp
        else:
            pass
    temp_stim_normal = temp_stim.plane.direction

    # face areas need to be area of face that is perpendicular (or close to) that of the temp bc in
    # environment
    # if no face geometry exists - calculate face area from edge and thickness, identifying edge as
    # being perpendicular to the normal of the stimulus
    face_areas = np.zeros(len(organism.faces))
    for k, f in organism.faces.items():
        face_normal = f.normal
        if np.dot(temp_stim_normal, face_normal) == 1.0:
            if f.surface_area != 0.0:
                face_areas[k] = f.surface_area
        else:
            for _ , edge in enumerate(f.edges):
                if np.dot(temp_stim_normal, edge.line.direction) == 0.0:
                    face_areas[k] = f.thickness * edge.length

    total_heat_transfer = np.sum(heat_flux_results * face_areas)

    return 1/total_heat_transfer

def _get_face_element_values(section_lines, out_array):

    this_element_id = int(section_lines[0].split(" ")[0])
    ind = 0

    for line in section_lines:
        data = line.split(" ")
        [data.remove(item) for item in data if len(item) == 0]
        [data.remove(item) for item in data if len(item) == 0]

        element_id = int(data[0])

        if element_id > this_element_id:
            out_array[ind] /= int_point
            ind += 1
            this_element_id = element_id

        int_point = int(data[1])
        int_pt_data = [float(i) for i in data[2:-1]]
        out_array[ind] += int_pt_data

    out_array[ind] /= int_point

    return out_array


def get_max_deflection(face_displacements):
    """Return the maximum displacmenent of the organism

    - To calculate the max deflection, the maximum displacement value in any axis for all the nodes
    is found.
    
    Args: 
        face_displacements: dict with face id as key and list of node displacements of that face

    Returns
        max_displacment: float of max nodal displacement 
    """

    if isinstance(face_displacements, np.ndarray):
        return np.max(np.abs(face_displacements))
    

def get_node_temp(points, fit_criteria, node_temperatures):
    """Return the temperature of the node nearest the point attribute of the stimulus
    
    Args:
        fit_criteria: fitness criteria data class
        node_temperature: dictionary of nodes and their temperatures from ccx .dat file
    """
    fitness = 1e9

    for node_id, node_temp in node_temperatures.items():

        if np.allclose(points[node_id-1].get_point_array(), np.array(astuple(fit_criteria))):
            fitness = node_temp

    return fitness

def get_average_node_temp(points, fit_criteria, node_temperatures):
    """Return the average temperature of the nodes (considering the constraint in the stimulus)
    
    Args:
        fit_criteria: fitness criteria data class
        node_temperature: dictionary of nodes and their temperatures from ccx .dat file
    """
    total_node_temp = []

    fitness = 1e9

    for node_id, node_temp in node_temperatures.items():
        total_node_temp.append(node_temp)

    average_node_temp = sum(total_node_temp) / len(total_node_temp)

    if fit_criteria.target is not None:
        fitness = abs(fit_criteria.target - average_node_temp)
    else:
        fitness = average_node_temp

    return fitness

def get_temperature_difference(organism, fit_criteria, node_tempertures):
    """
    Get the temperature difference between two nodes in the model
    """
    pt1 = fit_criteria.point1
    pt2 = fit_criteria.point2

    for id, point in organism.points.items():
        if np.allclose(pt1.as_array, point.as_array):
            node1 = id + 1
        elif np.allclose(pt2.as_array, point.as_array):
            node2 = id + 1

    temp_diff = node_tempertures[node1] -  node_tempertures[node2]

    return temp_diff

def get_volume_fraction(organism):
    """Get the fraction of edge volume to cell volume of the organism
    
    Args:
        organism: organism object containing geometry

    Returns:
        Float of volume fraction of organism
    """
    total_cell_volume = get_total_volume_dim_two(organism)
    edges = organism.edges
    line_lengths = []
    edge_areas = []
    for _, edge in sorted(edges.items()):
        line_lengths.append(edge.line.length)
        edge_areas.append(edge.cs_area)
    total_edge_volume = get_total_volume(np.asarray(line_lengths), np.asarray(edge_areas))

    return total_edge_volume / total_cell_volume

def get_edge_volume(organism):
    """Get the edge volume of the organism
    
    Args:
        organism: organism object containing geometry

    Returns:
        Float of edge volume of organism
    """
    edges = organism.edges
    edge_volumes = 0
    for _, edge in sorted(edges.items()):
        vol = edge.volume
        edge_volumes += vol   

    return edge_volumes


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
        organism: organism object

    Returns:
        A single total volume float.
    """

    # Face Volume
    face_areas = []
    thicknesses = []

    faces = organism.faces
    for _ , face in sorted(faces.items()):
        if not face.virtual:
            face_areas.append(face.surface_area)
            thicknesses.append(face.thickness)
    face_areas = np.asarray(face_areas)
    thicknesses = np.asarray(thicknesses)
    face_volume = np.sum(face_areas * thicknesses)
    # Edge Volume
    edge_volume = 0
    edges = organism.edges
    for _, edge in sorted(edges.items()):
        edge_volume += np.asarray(edge.volume)

    return face_volume + edge_volume


def get_total_face_area(organism=None):
    """Get the total area of the organism.

    Args:


    Returns:
        A single total volume float.
    """
    face_areas = []
    faces = organism.faces
    for _ , face in sorted(faces.items()):
        if not face.virtual:
            face_areas.append(face.surface_area)
    face_areas = np.asarray(face_areas)
    return np.sum(face_areas)    

def get_non_dim_temp(node_temperatures, environment, organism):
    """Get the non-dimensional temperature of the organism

    Args:
        node_temperatures: temperatures of the organism
        organism: organism object
    
    Returns:
        Non-dimensional temperature of the organism
    """
    Tmax = max(node_temperatures.values())
    Tmin = min(node_temperatures.values())
    k = organism.edges[0].material.conductivity[0]
    for stimulus in environment.stimuli:
        if isinstance(stimulus, stim.DistributedHeatFlux):
            heat_flux = stimulus.magnitude
            break
    Atot = get_total_face_area(organism)
    nd_temp = ((Tmax - Tmin) * k) / (heat_flux * Atot)

    return nd_temp

if __name__ == "__main__":
    print("End")
