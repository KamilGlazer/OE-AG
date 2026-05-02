def fitness_martin_gaddy(x):
    n = len(x)
    if n == 1:
        return ((x[0] - 5)/3)**2
    
    val = 0.0
    for i in range(n - 1):
        term1 = (x[i] - x[i+1])**2
        term2 = ((x[i] + x[i+1] - 10) / 3)**2
        val += term1 + term2
    return val

AVAILABLE_FUNCTIONS = {
    "Martin and Gaddy": fitness_martin_gaddy
}
