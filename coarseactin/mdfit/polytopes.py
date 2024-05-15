import numpy as np
import itertools
from scipy.spatial.transform import Rotation

def generate_600_vertices():
    ''' 120-cell vertices'''
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    vertices = set()

    # 24-cell vertices: permutations of {0, 0, ±2, ±2}
    for signs in itertools.product([2, -2], repeat=2):
        for permutation in itertools.permutations([0, 0] + list(signs), 4):
            vertices.add(tuple(permutation))

    # Three sets of 64 vertices each for the 120-cell
    sets = [
        [phi, phi, phi, phi-2],        # 64 {±φ, ±φ, ±φ, ±φ−2}
        [1, 1, 1, np.sqrt(5)],         # 64 {±1, ±1, ±1, ±√5}
        [phi-1, phi-1, phi-1, phi**2]  # 64 {±φ−1, ±φ−1, ±φ−1, ±φ2}
    ]
    for i, base_set in enumerate(sets, start=1):
        for signs in itertools.product([1, -1], repeat=4):
            for permutation in itertools.permutations([base_set[0]*signs[0], base_set[1]*signs[1], base_set[2]*signs[2], base_set[3]*signs[3]], 4):
                vertices.add(tuple(permutation))

    # Even permutations
    even_sets = [
        [0, phi-1, phi, np.sqrt(5)],  # [0, ±φ−1, ±φ, ±√5] 
        [0, phi-2, 1, phi**2],        # [0, ±φ−2, ±1, ±φ2]
        [phi-1, 1, phi, 2]            # [±φ−1, ±1, ±φ, ±2]
    ]
    for even_set in even_sets:
        for permutation in itertools.permutations(list(range(4)), 4):
            if not is_even_permutation(permutation):
                continue
            for signs in itertools.product([1,-1], repeat=4):
                seq = [signs[p]*even_set[p] for i,p in enumerate(permutation)]
                #Continue if 0 is multiplied by -1
                if signs[0] == -1 and even_set[0]==0:
                    continue
                vertices.add(tuple(seq))
        


    # Convert set to list and adjust for unit radius
    vertices = list(vertices)
    vertices = np.array(vertices)
    vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]

    return vertices

def generate_120_vertices():
    ''' 600-cell vertices'''
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Vertices set
    vertices = []

    # (0, 0, 0, ±1) and permutations thereof
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        vertices.append(e)
        vertices.append(-e)

    #print(len(vertices))
    # (±1/2, ±1/2, ±1/2, ±1/2)
    for signs in itertools.product(*[[-0.5, 0.5]]*4):
        vertices.append(np.array(signs))

    #print(len(vertices))

    # (±φ/2, ±1/2, ±1/2φ , 0) and permutations
    base_set = [phi/2, 1/2, 1/2/phi, 0]
    for permutation in itertools.permutations(list(range(4)), 4):
        if not is_even_permutation(permutation):
            continue
        for signs in itertools.product([1,-1], repeat=4):
            seq = [signs[p]*base_set[p] for i,p in enumerate(permutation)]
            #Continue if 0 is multiplied by -1
            if signs[3] == -1:
                continue

            vertices.append(np.array(seq))
        #print(len(vertices))
    
    # Normalize the vertices to get unit quaternions
    vertices = np.array(vertices)
    vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]

    return vertices

def generate_24_vertices():
    ''' 24-cell vertices'''
    vertices = []

    # Permutations of (±1, ±1, 0, 0)
    for signs in itertools.product([1/np.sqrt(2), -1/np.sqrt(2)], repeat=2):
        for zeros in itertools.combinations(range(4), 2):
            vertex = np.zeros(4)
            for i, sign in zip(zeros, signs):
                vertex[i] = sign
            vertices.append(vertex)
            vertices.append(-vertex)

    # Ensure uniqueness
    vertices = np.unique(vertices, axis=0)
    return vertices

def generate_16_vertices():
    ''' 8-cell vertices'''
    # Generating vertices for an 8-cell or hyper-cube
    vertices = np.array(list(itertools.product([-.5, .5], repeat=4)))
    return vertices


def generate_8_vertices():
    ''' 16-cell vertices'''
    # Correct approach to generate 16-cell vertices
    vertices = []
    # All permutations of (0, 0, 0, ±1) with each position being ±1 once
    for i in range(4):
        for sign in [1, -1]:
            vertex = np.zeros(4)
            vertex[i] = sign
            vertices.append(vertex)
    return np.array(vertices)

def generate_5_vertices():
    ''' 5-cell vertices'''
    
    s5 = np.sqrt(5)
    # Initial vertex
    vertices = np.array([[0, 0, 0, 4],
                         [s5, s5, s5, -1],
                         [s5, -s5, -s5, -1],
                         [-s5, s5, -s5, -1],
                         [-s5, -s5, s5, -1]])/4
    
    return vertices

def generate_random_vertices(n):
    ''' Generate n random vertices in 4D space.'''
    vertices = np.random.randn(n, 4)-0.5
    vertices /= np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    return vertices

def normalize_quaternions(quaternions):
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions / norms

def compute_repulsion_forces(quaternions):
    n = len(quaternions)
    # Expand quaternions into two matrices to compute pairwise differences
    q_expanded = np.expand_dims(quaternions, axis=1)  # Shape (n, 1, 4)
    q_tiled = np.tile(q_expanded, (1, n, 1))  # Shape (n, n, 4)
    q_transposed = np.transpose(q_tiled, (1, 0, 2))  # Shape (n, n, 4)

    # Compute pairwise differences
    diff = q_expanded - q_transposed  # Shape (n, n, 4)
    shadow_diff = q_expanded + q_transposed  # Shape (n, n, 4)
    diff = np.concatenate([diff, shadow_diff], axis=1)

    # Compute squared distances with a small epsilon to avoid division by zero
    dist2 = np.sum(diff**2, axis=2) + 1E-6  # Shape (n, 2n)

    # Compute force magnitudes (inverse square law)
    force_magnitude = 1 / dist2  # Shape (n, 2n)

    # Force vectors
    force_vectors = diff * np.expand_dims(force_magnitude, axis=2)  # Shape (n, 2n, 4)

    # Calculate the component of each force vector that is parallel to each quaternion
    parallel_components = np.sum(force_vectors * q_expanded, axis=2, keepdims=True) * q_expanded  # Shape (n, 2n, 4)

    # Calculate perpendicular forces by subtracting the parallel component
    perpendicular_forces = force_vectors - parallel_components  # Shape (n, 2n, 4)

    # Sum up all perpendicular forces for each quaternion
    total_forces = np.sum(perpendicular_forces, axis=1)  # Shape (n, 4)

    return total_forces

def optimize_quaternions(n, iterations=10000, initial_learning_rate=10000):
    # Initialize random quaternions
    quaternions = np.random.randn(n, 4)
    quaternions = normalize_quaternions(quaternions)
    converged=False
    learning_rate = initial_learning_rate
    for iteration in range(iterations):
        forces = compute_repulsion_forces(quaternions)

        # Update quaternion positions
        quaternions += learning_rate * forces

        # Renormalize to ensure they stay on the 4D unit hypersphere
        quaternions = normalize_quaternions(quaternions)

        # Reduce learning rate over time
        learning_rate = initial_learning_rate /n / (1 + 1.5 * iteration)

        # Early stopping condition if forces are sufficiently small
        if np.max(np.linalg.norm(forces, axis=1)) < 1e-6:
            print('Converged', iteration)
            converged=True
            break
        
    if not converged:
        print('Not Converged', np.max(np.linalg.norm(forces, axis=1)))

    return quaternions

def generate_rotations(n, optimize=0):
    """
    Generate a set of rotations represented by quaternions. If the specified number of rotations corresponds
    to a regular polytope, the vertices of that polytope are used. Otherwise, generates n random rotations and
    optionally optimizes them for even spacing.

    Regular polytope numbers and their corresponding vertices are:
    - 300: 600-cell vertices
    - 60: 120-cell vertices
    - 12: 24-cell vertices
    - 8: 16-cell vertices
    - 5: 5-cell vertices
    - 4: 8-cell vertices

    Parameters
    ----------
    n : int
        The number of rotations to generate.
    optimize : int, optional
        Number of iterations for optimizing the distribution of rotations if n does not correspond to a regular polytope.
        If optimize is 0, no optimization is performed. Default is 0.

    Returns
    -------
    rotations : Rotation
        A scipy.spatial.transform.Rotation object representing the rotations generated. Rotations are based on unit quaternions.
    """
    # Dictionary to map numbers to specific polytope vertex generators
    polytope_generators = {
        300: generate_600_vertices,
        60: generate_120_vertices,
        12: generate_24_vertices,
        8: generate_16_vertices,
        5: generate_5_vertices,
        4: generate_8_vertices,
    }

    # Check if the number of vertices can be generated from regular polytopes
    if n in polytope_generators:
        vertices = polytope_generators[n]()
        unique_quaternions = set()
        for q in vertices:
            rounded_q = tuple(np.round(q, decimals=8))
            neg_rounded_q = tuple(np.round(-q, decimals=8))
            if rounded_q not in unique_quaternions and neg_rounded_q not in unique_quaternions:
                unique_quaternions.add(rounded_q)
        vertices = list(unique_quaternions)
        if optimize:
            # Optimize these vertices to spread them evenly
            print('Vertices will not be optimized for regular polytopes')
    else:   
        # Generate n random vertices
        vertices = generate_random_vertices(n)
        if optimize:
            # Optimize these vertices to spread them evenly
            vertices = optimize_quaternions(n, iterations=optimize)

    # Normalize the vertices to make sure they are unit quaternions
    vertices = normalize_quaternions(vertices)

    # Convert the unit quaternions to rotations
    rotations = Rotation.from_quat(vertices)
    return rotations

# # Generate and normalize vertices
# vertices = generate_24cell_vertices()
# vertices = vertices / np.linalg.norm(vertices, axis=1)[:, None]

# # Unique quaternions, handling floating-point precision
# unique_quaternions = set()
# for q in vertices:
#     rounded_q = tuple(np.round(q, decimals=8))
#     neg_rounded_q = tuple(np.round(-q, decimals=8))
#     if rounded_q not in unique_quaternions and neg_rounded_q not in unique_quaternions:
#         unique_quaternions.add(rounded_q)

# vertices2 = list(unique_quaternions)
# vertices2.sort()

# # Convert to rotations
# rotations = R.from_quat(vertices2)



def is_even_permutation(permutation):
    '''Determine the parity of a permutation.'''
    n = len(permutation)
    seen = set()
    swaps = 0
    for i in range(n):
        if i not in seen:
            j = i
            while permutation[j] not in seen:
                seen.add(j)
                j = permutation[j]
                swaps += 1
            swaps -= 1  
    return swaps % 2 == 0

def is_equidistant(vertices):
    distances = []
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(vertices[i] - vertices[j])
            distances.append(dist)
            #print(i,j,dist)
    
    if np.std(distances) < 1e-9:
        return True
    else:
        return False
    
def is_regular_polytope(vertices):
    n = len(vertices)
    # Assuming a 3D cube to find 3 nearest neighbors as an example, adjust 'expected_neighbors' as needed
    expected_neighbors = None  # In a regular polytope, expect 'dim' nearest neighbors
    regularity_issues = []
    regular=True
    expected_distance = None
    for i in range(n):
        distances = np.round([np.linalg.norm(vertices[i] - vertices[j]) for j in range(n) if i != j],5)
        min_distance = min(distances)
        if not expected_distance:
            expected_distance = min_distance
        count_min_distance = list(distances).count(expected_distance)
        #print(i,min_distance,expected_distance)
        # Initialize expected_neighbors with the count of the first vertex's nearest neighbors
        if expected_neighbors is None:
            expected_neighbors = count_min_distance
        
        # If any vertex does not match the expected pattern, print detailed information
        elif count_min_distance != expected_neighbors:
            # Find all vertices that are at the min_distance from vertex i
            nearest_neighbors = [j for j in range(n) if i != j and np.isclose(np.linalg.norm(vertices[i] - vertices[j]), min_distance, atol=1e-5)]
            expected_neighbor_list = [j for j in range(n) if i != j and np.isclose(np.linalg.norm(vertices[i] - vertices[j]), expected_distance, atol=1e-5)]
            issue = f"Vertex {i} FAILS the regularity test: {count_min_distance} nearest neighbors at distance {expected_distance}, expected {expected_neighbors}.\n"
            issue +=f"{expected_neighbor_list}\n"
            issue +=f"Neighbors at {min_distance}: {nearest_neighbors}\n"
            regularity_issues.append(issue)
            regular=False
    
    # Report all regularity issues
    for issue in regularity_issues:
        print(issue)
    
    return regular
    
def is_unit_vector(vertices):
    distances = []
    n = len(vertices)
    for i in range(n):
        dist = np.linalg.norm(vertices[i])
        distances.append(dist)
    distances=np.array(distances)
    #print(np.unique(distances))
    if np.sqrt(((distances-1)**2).sum()) < 1e-9:
        return True
    else:
        return False





# Test Execution for All Polytopes
def test_all_polytopes_corrected():
    results = {}
    polytope_generators = {
        600: generate_600_vertices,
        120: generate_120_vertices,
        24: generate_24_vertices,
        16: generate_16_vertices,
        8: generate_8_vertices,
        5: generate_5_vertices,
        121: lambda: generate_random_vertices(n=121),
    }
    
    for n, generator in polytope_generators.items():
        vertices = generator()
        results['Regular: '+str(n)] = is_regular_polytope(vertices)
        results['UnitVector: '+str(n)] = is_unit_vector(vertices)
        results['Quantity: '+str(n)] = len(vertices)==n
        #for i,v in enumerate(vertices):
            #print(i,v)
    return results

if __name__ == '__main__':
    results = test_all_polytopes_corrected()
    for name, result in results.items():
        print(f'{name}-cell: {result}')
    generate_rotations(500, optimize=0)
    generate_rotations(200, optimize=1)
    generate_rotations(7, optimize=2000)
    generate_rotations(300, optimize=10)
    generate_rotations(60, optimize=10)

    for n in [300, 60, 24, 16, 8, 5, 121]:
        rotations = generate_rotations(n, optimize=0)
        print(f'{n}: {len(rotations.as_quat())}')
