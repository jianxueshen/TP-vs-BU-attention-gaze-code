import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os


def generateStructureArrayScanpath(data):
    len_data = data.shape[0]
    sp1 = {
        'fixation': {
            'x': np.zeros(len_data),
            'y': np.zeros(len_data),
            'dur': np.zeros(len_data)
        },
        'saccade': {
            'x': np.zeros(len_data-1),
            'y': np.zeros(len_data-1),
            'lenx': np.zeros(len_data-1),
            'leny': np.zeros(len_data-1),
            'theta': np.zeros(len_data-1),
            'len': np.zeros(len_data-1)
        }
    }

    for i in range(len_data):
        sp1['fixation']['x'][i] = data.loc[i, 'Start X']
        sp1['fixation']['y'][i] = data.loc[i, 'Start Y']
        sp1['fixation']['dur'][i] = data.loc[i, 'Duration ms']

        if i < len_data - 1:
            sp1['saccade']['x'][i] = data.loc[i, 'Start X']
            sp1['saccade']['y'][i] = data.loc[i, 'Start Y']

        if i > 0:
            sp1['saccade']['lenx'][i - 1] = sp1['fixation']['x'][i] - sp1['saccade']['x'][i - 1]
            sp1['saccade']['leny'][i - 1] = sp1['fixation']['y'][i] - sp1['saccade']['y'][i - 1]
            [sp1['saccade']['theta'][i - 1], sp1['saccade']['len'][i - 1]] = cart2pol(sp1['saccade']['lenx'][i - 1], sp1['saccade']['leny'][i - 1])

    return sp1
#计算扫视向量差异
def calVectorDifferences(x1, x2, y1, y2): 
    # Initialize the matrix M with zeros
    M = np.zeros((len(x1), len(x2)))

    # Calculate vector differences
    for k in range(len(x1)):
        # Create arrays with repeated x1(k) and y1(k) values
        x_diff = np.abs(x1[k] - x2)
        y_diff = np.abs(y1[k] - y2)
        
        # Calculate the Euclidean distance for each pair
        M[k, :] = np.sqrt(x_diff**2 + y_diff**2)

    return M
#计算扫视角度差异
def calAngularDifferences(theta1, theta2, path, M_assignment):
        angle_diff = np.zeros(len(path))

        # Compare scanpaths
        for k in range(len(path)):
            # Current path element is expected to be a tuple (i, j)
            current_path_element = path[k]

            # Check if current_path_element is valid for M_assignment
            if current_path_element in M_assignment:
                i, j = current_path_element  # Directly unpack the tuple

                # Calculate the absolute difference between angles
                spT_diff = np.abs(theta1[i] - theta2[j])

                # Adjust angles that are greater than 180 degrees
                if spT_diff > 180:
                    spT_diff = 360 - spT_diff

                # Store the result
                angle_diff[k] = spT_diff
            else:
                print(f"No valid indices found for path element {current_path_element}.")
                angle_diff[k] = np.nan  # Assign NaN or some default value for invalid indices

        return angle_diff
#计算持续时间差异
def calDurationDifferences(dur1, dur2, path, M_assignment):
    dur_diff = np.zeros(len(path))

    # Compare scanpaths
    for k in range(len(path)):
        # Current path element is expected to be a tuple (i, j)
        current_path_element = path[k]

        # Check if current_path_element is valid for M_assignment
        if current_path_element in M_assignment:
            i, j = current_path_element  # Directly unpack the tuple

            # Calculate the absolute difference and normalize by the maximum duration
            max_duration = max(dur1[i], dur2[j])
            if max_duration > 0:  # Prevent division by zero
                dur_diff[k] = np.abs(dur1[i] - dur2[j]) / max_duration
            else:
                dur_diff[k] = 0  # Handle case where both durations are zero
        else:
            print(f"No valid indices found for path element {current_path_element}.")
            dur_diff[k] = np.nan  # Assign NaN or some default value for invalid indices

    return dur_diff
#计算扫视长度差异
def calLengthDifferences(len1, len2, path, M_assignment):
    length_diff = np.zeros(len(path))

    # Compare scanpaths
    for k in range(len(path)):
        # Current path element is expected to be a tuple (i, j)
        current_path_element = path[k]

        # Check if current_path_element is valid for M_assignment
        if current_path_element in M_assignment:
            i, j = current_path_element  # Directly unpack the tuple

            # Calculate the absolute difference in length
            length_diff[k] = np.abs(len1[i] - len2[j])
        else:
            print(f"No valid indices found for path element {current_path_element}.")
            length_diff[k] = np.nan  # Assign NaN or some default value for invalid indices

    return length_diff
#计算注视点距离差异
def calPositionDifferences(x1, x2, y1, y2, path, M_assignment):
    pos_diff = np.zeros(len(path))

    # Compare scanpaths
    for k in range(len(path)):
        # Current path element is expected to be a tuple (i, j)
        current_path_element = path[k]

        # Check if current_path_element is valid for M_assignment
        if current_path_element in M_assignment:
            i, j = current_path_element  # Directly unpack the tuple

            # Calculate the Euclidean distance between points
            pos_diff[k] = np.sqrt((x1[i] - x2[j])**2 + (y1[i] - y2[j])**2)
        else:
            print(f"No valid indices found for path element {current_path_element}.")
            pos_diff[k] = np.nan  # Assign NaN or some default value for invalid indices

    return pos_diff

def calVectorDifferencesAlongPath(x1, x2, y1, y2, path, M_assignment):
    # Initialize a 2D array to hold the differences
    vector_diff = np.zeros((len(path), 2))

    # Compare scanpaths along the specified path
    for k in range(len(path)):
        # Current path element is expected to be a tuple (i, j)
        current_path_element = path[k]

        # Check if current_path_element is valid for M_assignment
        if current_path_element in M_assignment:
            i, j = current_path_element

            # Calculate the differences for x and y components
            vector_diff[k, 0] = x1[i] - x2[j]  # x component difference
            vector_diff[k, 1] = y1[i] - y2[j]  # y component difference
        else:
            print(f"No valid indices found for path element {current_path_element}.")
            continue

    return vector_diff


"""
def shortestPath(M):
    global imwritePath, fontSize, debugMode

    szM = M.shape
    AM = np.zeros((szM[0] * szM[1], szM[0] * szM[1]))
    M_assignment = np.zeros((szM[0], szM[1]))

    # Create adjacency matrix
    for i in range(szM[0]):
        for j in range(szM[1]):
            currentNode = (i * szM[1] + j)
            M_assignment[i, j] = currentNode
            i_inc = 1
            j_inc = 1

            if i == szM[0] - 1 and j < szM[1] - 1:
                i_inc = 0
                node_adjacent_ids = [currentNode, currentNode + 1]  # To the right
                node_weights = [M[i, j + j_inc]]
                AM[currentNode, currentNode + 1] = 1

            elif i < szM[0] - 1 and j == szM[1] - 1:
                j_inc = 0
                node_adjacent_ids = [currentNode, currentNode + szM[1]]  # Below
                node_weights = [M[i + i_inc, j]]
                AM[currentNode, currentNode + szM[1]] = 1

            elif i == szM[0] - 1 and j == szM[1] - 1:
                i_inc = 0
                j_inc = 0
                node_adjacent_ids = [currentNode, currentNode]
                node_weights = [0]

            else:
                node_adjacent_ids = [currentNode, currentNode + 1, currentNode + szM[1], currentNode + szM[1] + 1]
                node_weights = [M[i, j + j_inc], M[i + i_inc, j], M[i + i_inc, j + j_inc]]
                AM[currentNode, currentNode + 1] = 1
                AM[currentNode, currentNode + szM[1]] = 1
                AM[currentNode, currentNode + szM[1] + 1] = 1

            # Assuming node is a dictionary with 'adjacentNr' and 'weight' keys
            node = {'adjacentNr': node_adjacent_ids, 'weight': node_weights}

    # Create the graph
    G = nx.Graph()
    for i in range(szM[0]):
        for j in range(szM[1]):
            G.add_node((i, j), **node)
    
    if debugMode == 1:
        plt.figure()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=12)
        plt.show()
        plt.pause(2)
        plt.savefig(imwritePath + '/Graph.pdf')
        plt.close()
    

    # Calculate the shortest path
    dist, path = nx.single_source_dijkstra(G, (0, 0), (szM[0] - 1, szM[1] - 1))

    if debugMode == 1:
        plt.figure()
        pos = nx.spring_layout
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=12)
        nx.draw_networkx_edges(G, pos, edgelist=path, edge_color='red', width=2)
        plt.show()
        plt.pause(2)
        plt.savefig(imwritePath + '/GraphShortestPath.pdf')
        plt.close()

    return dist, path, M_assignment
"""
#最短路径算法
def shortestPath(M):
    global imwritePath, fontSize, debugMode

    szM = M.shape
    M_assignment = {} 
    
    if szM[0] <= 0 or szM[1] <= 0:
        print("Error: The input matrix M must have positive dimensions.")
        return None, None, M_assignment


    # Create the graph
    G = nx.Graph()

    # Add nodes and edges to the graph
    for i in range(szM[0]):
        for j in range(szM[1]):
            currentNode = (i, j)
            M_assignment[(i, j)] = (i, j)

            # Connect to the right
            if j < szM[1] - 1:
                rightNode = (i, j + 1)
                G.add_edge(currentNode, rightNode, weight=M[i, j + 1])

            # Connect below
            if i < szM[0] - 1:
                belowNode = (i + 1, j)
                G.add_edge(currentNode, belowNode, weight=M[i + 1, j])

            # Connect diagonally right and below
            if i < szM[0] - 1 and j < szM[1] - 1:
                diagNode = (i + 1, j + 1)
                G.add_edge(currentNode, diagNode, weight=M[i + 1, j + 1])


    pos = nx.spring_layout(G)
    # Calculate the shortest path
    start_node = (0, 0)
    end_node = (szM[0] - 1, szM[1] - 1)
    
    if start_node not in G.nodes or end_node not in G.nodes:
        print("Start or end node not found in the graph.")
        return None, None, M_assignment

    try:
        path = nx.dijkstra_path(G, start_node, end_node)
        dist = nx.path_weight(G, path, weight='weight')
    except nx.NetworkXNoPath:
        print("No path exists between the start and end nodes.")
        return None, None, M_assignment
    """
    # Visualize the shortest path
    plt.figure(figsize=(15, 12))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=10, font_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:-1], path[1:])), edge_color='red', width=2)
    #plt.show()
    #plt.pause(2)
    plt.savefig(save_path_shortest)
    plt.close()
    """
    return dist, path, M_assignment


def cart2pol(x, y):
    r = np.sqrt(x**2 + y**2)
    theta_radians = np.arctan2(y, x)
    theta_degrees = np.degrees(theta_radians)
    return theta_degrees, r

def keepSaccade(sp, spGlobal, i, durMem):
    """
    Updates spGlobal with the current saccade data and manages the duration memory.

    Parameters:
    sp (dict): The original scanpath containing saccade and fixation data.
    spGlobal (dict): The global scanpath to which data will be added.
    i (int): The current index in the original scanpath.
    j (int): The current index for appending in the global scanpath.
    durMem (float): The duration memory to be added to the fixation duration.

    Returns:
    tuple: Updated sp, spGlobal, new index i, and updated durMem.
    """
    
    # Append the current saccade data to spGlobal
    spGlobal['saccade']['x'] = np.append(spGlobal['saccade']['x'], sp['saccade']['x'][i])
    spGlobal['saccade']['y'] = np.append(spGlobal['saccade']['y'], sp['saccade']['y'][i])
    spGlobal['saccade']['lenx'] = np.append(spGlobal['saccade']['lenx'], sp['saccade']['lenx'][i])
    spGlobal['saccade']['leny'] = np.append(spGlobal['saccade']['leny'], sp['saccade']['leny'][i])
    spGlobal['saccade']['len'] = np.append(spGlobal['saccade']['len'], sp['saccade']['len'][i])
    spGlobal['saccade']['theta'] = np.append(spGlobal['saccade']['theta'], sp['saccade']['theta'][i])

    # Append fixation data
    spGlobal['fixation']['x'] = np.append(spGlobal['fixation']['x'], sp['fixation']['x'][i])
    spGlobal['fixation']['y'] = np.append(spGlobal['fixation']['y'], sp['fixation']['y'][i])
    spGlobal['fixation']['dur'] = np.append(spGlobal['fixation']['dur'], sp['fixation']['dur'][i] + durMem)
    
    # Reset durMem to 0
    durMem = 0

    # Move to the next saccade in the original scanpath
    i += 1
    return sp, spGlobal, i, durMem


def simplifyDirection(sp, T, Tdur):
    """
    Merges a series of short saccades
    
    Input:    sp - scanpath vectors
              T - threshold separating local from global scanpath
    Output:   sp_out - simplified scanpath

    Suggested modifications: Change 'and' to 'or'?
    """

    spGlobal = {
        'fixation': {
            'x': np.array([]),
            'y': np.array([]),
            'dur': np.array([])
        },
        'saccade': {
            'x': np.array([]),
            'y': np.array([]),
            'lenx': np.array([]),
            'leny': np.array([]),
            'theta': np.array([]),
            'len': np.array([])
        }
    }

    if sp['saccade']['x'].shape[0] <= 1:
        return sp

    i = 0
    durMem = 0

    while i < sp['saccade']['x'].shape[0]-1:
        if i + 1 >= sp['saccade']['x'].shape[0]:
            break

        v1 = [sp['saccade']['lenx'][i], sp['saccade']['leny'][i]]
        v2 = [sp['saccade']['lenx'][i + 1], sp['saccade']['leny'][i + 1]]
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

        if angle < T:
            if sp['fixation']['dur'][i + 1] >= Tdur:
                sp, spGlobal, i, durMem = keepSaccade(sp, spGlobal, i, durMem)
                continue
            
            v_x = sp['saccade']['lenx'][i] + sp['saccade']['lenx'][i + 1]
            v_y = sp['saccade']['leny'][i] + sp['saccade']['leny'][i + 1]
            
            [theta, len] = cart2pol(v_x, v_y)

            spGlobal['saccade']['x'] = np.append(spGlobal['saccade']['x'], sp['saccade']['x'][i])
            spGlobal['saccade']['y'] = np.append(spGlobal['saccade']['y'], sp['saccade']['y'][i])
            spGlobal['saccade']['lenx'] = np.append(spGlobal['saccade']['lenx'], sp['saccade']['lenx'][i])
            spGlobal['saccade']['leny'] = np.append(spGlobal['saccade']['leny'], sp['saccade']['leny'][i])
            spGlobal['saccade']['len'] = np.append(spGlobal['saccade']['len'], sp['saccade']['len'][i])
            spGlobal['saccade']['theta'] = np.append(spGlobal['saccade']['theta'], sp['saccade']['theta'][i])

            # Append fixation data
            spGlobal['fixation']['x'] = np.append(spGlobal['fixation']['x'], sp['fixation']['x'][i])
            spGlobal['fixation']['y'] = np.append(spGlobal['fixation']['y'], sp['fixation']['y'][i])
            spGlobal['fixation']['dur'] = np.append(spGlobal['fixation']['dur'], sp['fixation']['dur'][i] + durMem)
            durMem = 0
            i = i + 2

        else:
            sp, spGlobal, i, durMem = keepSaccade(sp, spGlobal, i, durMem)

    if i == sp['saccade']['x'].shape[0]-1:
        sp, spGlobal, i, durMem = keepSaccade(sp, spGlobal, i, durMem)
        spGlobal['fixation']['x'] = np.append(spGlobal['fixation']['x'], sp['fixation']['x'][-1])
        spGlobal['fixation']['y'] = np.append(spGlobal['fixation']['y'], sp['fixation']['y'][-1])
        spGlobal['fixation']['dur'] = np.append(spGlobal['fixation']['dur'], sp['fixation']['dur'][-1] + durMem)
    return spGlobal

def simplifyLength(sp, T, Tdur):
    
    spGlobal = {
        'fixation': {
            'x': np.array([]),
            'y': np.array([]),
            'dur': np.array([])
        },
        'saccade': {
            'x': np.array([]),
            'y': np.array([]),
            'lenx': np.array([]),
            'leny': np.array([]),
            'theta': np.array([]),
            'len': np.array([])
        }
    }

    if sp['saccade']['x'].shape[0] <= 1:
        return sp

    i = 0
    durMem = 0

    while i < sp['saccade']['x'].shape[0]-1:
        if sp['saccade']['len'][i] < T:
            if sp['fixation']['dur'][i] >= Tdur and sp['fixation']['dur'][i + 1] >= Tdur:
                sp, spGlobal, i, durMem = keepSaccade(sp, spGlobal, i, durMem)
                continue

            v_x = sp['saccade']['lenx'][i] + sp['saccade']['lenx'][i + 1]
            v_y = sp['saccade']['leny'][i] + sp['saccade']['leny'][i + 1]
            [theta, len] = cart2pol(v_x, v_y)

            spGlobal['saccade']['x'] = np.append(spGlobal['saccade']['x'], sp['saccade']['x'][i])
            spGlobal['saccade']['y'] = np.append(spGlobal['saccade']['y'], sp['saccade']['y'][i])
            spGlobal['saccade']['lenx'] = np.append(spGlobal['saccade']['lenx'], sp['saccade']['lenx'][i])
            spGlobal['saccade']['leny'] = np.append(spGlobal['saccade']['leny'], sp['saccade']['leny'][i])
            spGlobal['saccade']['len'] = np.append(spGlobal['saccade']['len'], sp['saccade']['len'][i])
            spGlobal['saccade']['theta'] = np.append(spGlobal['saccade']['theta'], sp['saccade']['theta'][i])

            # Append fixation data
            spGlobal['fixation']['x'] = np.append(spGlobal['fixation']['x'], sp['fixation']['x'][i])
            spGlobal['fixation']['y'] = np.append(spGlobal['fixation']['y'], sp['fixation']['y'][i])
            spGlobal['fixation']['dur'] = np.append(spGlobal['fixation']['dur'], sp['fixation']['dur'][i] + durMem)
            durMem = 0
            i = i + 2

        else:
            sp, spGlobal, i, durMem = keepSaccade(sp, spGlobal, i, durMem)

    if i == sp['saccade']['x'].shape[0]-1:
        sp, spGlobal, i, durMem = keepSaccade(sp, spGlobal, i, durMem)
        spGlobal['fixation']['x'] = np.append(spGlobal['fixation']['x'], sp['fixation']['x'][-1])
        spGlobal['fixation']['y'] = np.append(spGlobal['fixation']['y'], sp['fixation']['y'][-1])
        spGlobal['fixation']['dur'] = np.append(spGlobal['fixation']['dur'], sp['fixation']['dur'][-1] + durMem)
    return spGlobal

def simplifyScanpath(sp, T1, T2, Tdur):
    l = np.inf
    spGlobal = sp

    while True:
        spGlobal = simplifyDirection(spGlobal, T2, Tdur)
        spGlobal = simplifyLength(spGlobal, T1, Tdur)

        if l == len(spGlobal['saccade']['x']):
            break
        l = len(spGlobal['saccade']['x'])

    return spGlobal


def mainProposed(sp1, sp2, T1, T2, Tdur, simple=False):
    """
    Main function to simplify and compare scanpaths.
    
    Parameters:
    sp1: scanpath objects
    sp2: scanpath objects
    T1: Length threshold for simplification px
    T2: Direction threshold for simplification degree
    Tdur: Duration threshold for simplification ms
    sz: Size of the screen (width, height)

    Returns:
    sp1: Simplified scanpaths 1
    sp2: Simplified scanpaths 2
    results: Comparison results
    path: Aligned path
    M_assignment: Mapping between scanpaths

    """

    results = {}

    # Simplify scanpaths
    if simple:
        sp1 = simplifyScanpath(sp1, T1, T2, Tdur)
        sp2 = simplifyScanpath(sp2, T1, T2, Tdur)

    # Vector differences
    M = calVectorDifferences(sp1['saccade']['lenx'], sp2['saccade']['lenx'],
                             sp1['saccade']['leny'], sp2['saccade']['leny'])
    M += np.finfo(float).eps  # Add small value to avoid division by zero

    # Align the scanpaths and find the cost associated with the best alignment
    dist, path, M_assignment = shortestPath(M)
    
    if path is None:
        results['vector_x_diff_px'] = np.nan
        results['vector_y_diff_px'] = np.nan
        results['angular_diff_degree'] = np.nan
        results['length_diff_px'] = np.nan
        results['position_diff_px'] = np.nan
        results['duration_diff_ms'] = np.nan
    else:
        # Compare scanpaths for other dimensions (along the aligned path)
        vector_diff = calVectorDifferencesAlongPath(sp1['saccade']['lenx'],
                    sp2['saccade']['lenx'],
                    sp1['saccade']['leny'],
                    sp2['saccade']['leny'],
                    path, M_assignment)

        results['vector_x_diff_px'] = np.mean(vector_diff[:, 0])
        results['vector_y_diff_px'] = np.mean(vector_diff[:, 1])

        results['angular_diff_degree'] = np.median(calAngularDifferences(sp1['saccade']['theta'],
                                            sp2['saccade']['theta'],
                                            path, M_assignment))

        results['length_diff_px'] = np.median(calLengthDifferences(sp1['saccade']['len'],
                                        sp2['saccade']['len'],
                                        path, M_assignment))

        results['position_diff_px'] = np.median(calPositionDifferences(sp1['saccade']['x'],
                                            sp2['saccade']['x'],
                                            sp1['saccade']['y'],
                                            sp2['saccade']['y'],
                                            path, M_assignment))
        
        results['duration_diff_ms'] = np.median(calDurationDifferences(sp1['fixation']['dur'],
                                            sp2['fixation']['dur'],
                                            path, M_assignment))


    return sp1, sp2, results, path, M_assignment

def plotGazePaths(sp1, sp2, title,save_path):
    plt.figure(figsize=(18, 10))

    # Plot first gaze path (sp1)
    for i in range(len(sp1['fixation']['x'])):
        plt.scatter(sp1['fixation']['x'][i], sp1['fixation']['y'][i], 
                    s=sp1['fixation']['dur'][i] / 5,  # Scale the size of the circle by duration
                    facecolors='none', edgecolors='blue', alpha=0.5, label='Fixation 1' if i == 0 else "")
        plt.text(sp1['fixation']['x'][i], sp1['fixation']['y'][i], str(i), 
             fontsize=10, ha='center', va='center', color='black')

    for i in range(len(sp1['saccade']['x'])-1):
        plt.plot(
            [sp1['fixation']['x'][i], sp1['fixation']['x'][i+1]], 
            [sp1['fixation']['y'][i], sp1['fixation']['y'][i+1]], 
            color='blue', linestyle='solid', alpha=0.7, label='Saccade 1' if i == 0 else ""
        )

    # Plot second gaze path (sp2)
    for i in range(len(sp2['fixation']['x'])):
        plt.scatter(sp2['fixation']['x'][i], sp2['fixation']['y'][i], 
                    s=sp2['fixation']['dur'][i] / 10,  # Scale the size of the circle by duration
                    facecolors='none', edgecolors='red', alpha=0.5, label='Fixation 2' if i == 0 else "")
        plt.text(sp2['fixation']['x'][i], sp2['fixation']['y'][i], str(i), 
             fontsize=10, ha='center', va='center', color='black')

    for i in range(len(sp2['saccade']['x'])-1):
        plt.plot(
            [sp2['fixation']['x'][i], sp2['fixation']['x'][i+1]], 
            [sp2['fixation']['y'][i], sp2['fixation']['y'][i+1]], 
            color='red', linestyle='dashed', alpha=0.7, label='Saccade 2' if i == 0 else ""
        )

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    #plt.show()
    plt.savefig(save_path)
    plt.close() 


def doComparison(data1, data2):
    # Parameters for the proposed algorithm
    sz = [1920, 1080]  # Size from JOV experiment
    globalThreshold = 0.02 * np.sqrt(sz[0]**2 + sz[1]**2)  # Diagonal/10 ~ 5 degrees
    directionThreshold = 10
    durationThreshold = 200

    # Return NaN if scanpaths are too short
    if data1.shape[0] < 3 or data2.shape[0] < 3:
        return np.nan

    bx1 = (data1.loc[:,'Start X'] > 0) & (data1.loc[:,'Start X'] < sz[0])
    bx2 = (data2.loc[:,'Start X'] > 0) & (data2.loc[:,'Start X'] < sz[0])
    by1 = (data1.loc[:,'Start Y'] > 0) & (data1.loc[:,'Start Y'] < sz[1])
    by2 = (data2.loc[:,'Start Y'] > 0) & (data2.loc[:,'Start Y'] < sz[1])

    data1 = data1[bx1 & by1]
    data2 = data2[bx2 & by2]
    data1 = data1.reset_index()
    data2 = data2.reset_index()

    bx1 = (data1.loc[:,'Start X'] > 0) & (data1.loc[:,'Start X'] < sz[0])
    bx2 = (data2.loc[:,'Start X'] > 0) & (data2.loc[:,'Start X'] < sz[0])
    by1 = (data1.loc[:,'Start Y'] > 0) & (data1.loc[:,'Start Y'] < sz[1])
    by2 = (data2.loc[:,'Start Y'] > 0) & (data2.loc[:,'Start Y'] < sz[1])

    # Return NaN if any fixations go out of bounds
    if np.all(bx1) and np.all(bx2) and np.all(by1) and np.all(by2):
        # Transform into scanpath structure
        sp1 = generateStructureArrayScanpath(data1)
        sp2 = generateStructureArrayScanpath(data2)
        
        #p_plot_path = os.path.join(save_path,f"original_{title}.png")
        #plotGazePaths(sp1, sp2, title, p_plot_path)

        # Compare scanpaths using the proposed method
        #save_path_shortest = os.path.join(save_path, f"shorest_{title}.png")
        sp1, sp2, rv, path, M_assignment = mainProposed(sp1, sp2, globalThreshold, directionThreshold, durationThreshold)
        
        #sp_plot_path = os.path.join(save_path,f"simplied_{title}.png")
        #title_s = f"simpled_path_{title}"
        #plotGazePaths(sp1, sp2, title_s, sp_plot_path)
        # Print results to the prompt
        #print(f'Vector similarity = {rv[0]}')
        #print(f'Direction similarity = {rv[1]}')
        #print(f'Length similarity = {rv[2]}')
        #print(f'Position similarity = {rv[3]}')
        #print(f'Duration similarity = {rv[4]}')
        
        #column_names = ['Vector similarity', 'Direction similarity', 'Length similarity', 'Position similarity', 'Duration similarity']
        #df = pd.DataFrame([rv], columns=column_names)
        #csv_file_path = os.path.join(save_path,f"similary_{title}.csv")
        #df.to_csv(csv_file_path, index=False)
        return rv
    else:
        return np.nan
    

if __name__ == "__main__":
    """
    # Example 3x3 matrix with random weights
    M = np.array([
        [1, 3, 1],
        [2, 5, 2],
        [1, 1, 1]
    ])

    # You would also need to define the global variables that the function uses:
    imwritePath = '.'  # Current directory for saving images
    fontSize = 5      # Font size for labels in the graph
    debugMode = 1      # Set to 1 to enable debug mode and visualize the graph

    # Now you can call the shortestPath function with the matrix M
    # Make sure to have the function definition in your code as well
    dist, path, M_assignment = shortestPath(M)

    print("Shortest distance:", dist)
    print("Path:", path)
    """

    path1 = pd.read_csv("./PATH/UAV123_building2_PATH/UAV123_building2_PATH_FV/Dominant/UAV123_building2_subj_2_FV_PATH.csv")
    path1_fix = path1[path1["Type"] == 0]
    path2 = pd.read_csv("./PATH/UAV123_building2_PATH/UAV123_building2_PATH_TASK/Dominant/UAV123_building2_subj_2_TASK_PATH.csv")
    path2_fix = path2[path2["Type"] == 0]
    path1_fix = path1_fix.reset_index(drop=True)
    path2_fix = path2_fix.reset_index(drop=True)
    print(path1_fix.shape)
    print(path2_fix.shape)

    title = "Gaze Path Compare"
    save_path = "PATH_Compare_Plot"
    doComparison(path1_fix,path2_fix)