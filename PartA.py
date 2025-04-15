import numpy as np
import cmath 

def dot_properties(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    if v1.shape != v2.shape:
        raise ValueError("Must have same dimensions")
    
    dot_prod = np.dot(v1, np.conjugate(v2))

    magnitude = abs(dot_prod)

    real = dot_prod.real
    imaginary = dot_prod.imag

    phase = cmath.phase(dot_prod)

    print("Dot Product: ", dot_prod)
    print("Magnitude: ", magnitude)
    print("The Real part: ", real)
    print("The Imaginary part: ", imaginary)
    print("The Phase: ", phase)

    return dot_prod,magnitude,real,imaginary,phase

vec1 = [1 + 2j, 3 - 1j]
vec2 = [2 - 1j, 1 + 3j]

result = dot_properties(vec1, vec2)

vec3 = [1-4j, 1 + 1j]
vec4 = [5-4j, 3]

result2 = dot_properties(vec3,vec4)

