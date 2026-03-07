import math

def fitness_martin_gaddy(x):
    """
    Martin and Gaddy test function.
    Global minimum: x_i = 5 for all i, f(x) = 0
    Function extended to N-dimensions by summing over adjacent pairs.
    f(x) = sum_{i=0}^{N-2} [ (x_i - x_{i+1})^2 + ((x_i + x_{i+1} - 10)/3)^2 ]
    If N=1, we can just return ((x[0] - 5)/3)**2 
    """
    n = len(x)
    if n == 1:
        return ((x[0] - 5)/3)**2
    
    val = 0.0
    for i in range(n - 1):
        term1 = (x[i] - x[i+1])**2
        term2 = ((x[i] + x[i+1] - 10) / 3)**2
        val += term1 + term2
    return val

# A dictionary to easily add more functions to the GUI
AVAILABLE_FUNCTIONS = {
    "Martin and Gaddy": fitness_martin_gaddy
}
