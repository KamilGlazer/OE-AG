import random
import math
import copy

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome  # list of ints (0/1)
        self.real_values = []
        self.fitness = 0.0
        self.objective_val = 0.0

class GeneticAlgorithm:
    def __init__(self, fitness_func, num_vars, domain, precision_decimals, 
                 pop_size, epochs, prob_cross, prob_mut, prob_inv, elitism_count,
                 selection_method="Turniejowa", crossover_method="Jednopunktowe",
                 mutation_method="Jednopunktowa", opt_type="Min",
                 tournament_size=3, best_sel_percent=0.2):
        self.fitness_func = fitness_func
        self.num_vars = num_vars
        self.domain = domain
        self.precision_decimals = precision_decimals
        self.pop_size = pop_size
        self.epochs = epochs
        self.prob_cross = prob_cross
        self.prob_mut = prob_mut
        self.prob_inv = prob_inv
        self.elitism_count = elitism_count
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.opt_type = opt_type
        self.tournament_size = tournament_size
        self.best_sel_percent = best_sel_percent
        
        self.bits_per_var = self._calculate_bits()
        self.total_bits = self.bits_per_var * self.num_vars
        self.population = []
        
        # History
        self.best_history = []
        self.avg_history = []
        self.worst_history = []
        self.best_solution_ever = None

    def _calculate_bits(self):
        a, b = self.domain
        diff = b - a
        points = diff * (10 ** self.precision_decimals) + 1
        return math.ceil(math.log2(points))

    def _decode(self, chromosome):
        real_vals = []
        a, b = self.domain
        max_dec = (2 ** self.bits_per_var) - 1
        
        if max_dec == 0:
            return [a] * self.num_vars
            
        for i in range(self.num_vars):
            sub_chrom = chromosome[i*self.bits_per_var : (i+1)*self.bits_per_var]
            dec_val = int("".join(map(str, sub_chrom)), 2)
            real_val = a + dec_val * (b - a) / max_dec
            real_vals.append(real_val)
        return real_vals

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            chrom = [random.randint(0, 1) for _ in range(self.total_bits)]
            ind = Individual(chrom)
            self.population.append(ind)

    def evaluate_population(self):
        for ind in self.population:
            ind.real_values = self._decode(ind.chromosome)
            ind.objective_val = self.fitness_func(ind.real_values)
            
            # For minimization, we want lower objective, but fitness should be higher for better
            # Or we can just use objective_val directly depending on the selection algorithm
            ind.fitness = ind.objective_val

        # Sort population based on objective_val
        reverse_sort = True if self.opt_type == "Max" else False
        self.population.sort(key=lambda x: x.objective_val, reverse=reverse_sort)

        # Update best
        if self.best_solution_ever is None:
            self.best_solution_ever = copy.deepcopy(self.population[0])
        else:
            if self.opt_type == "Min" and self.population[0].objective_val < self.best_solution_ever.objective_val:
                self.best_solution_ever = copy.deepcopy(self.population[0])
            elif self.opt_type == "Max" and self.population[0].objective_val > self.best_solution_ever.objective_val:
                self.best_solution_ever = copy.deepcopy(self.population[0])

    def select(self):
        new_pop = []
        
        # Elitism
        if self.elitism_count > 0:
            for i in range(min(self.elitism_count, self.pop_size)):
                new_pop.append(copy.deepcopy(self.population[i]))
                
        while len(new_pop) < self.pop_size:
            if self.selection_method == "Najlepszych":
                pool_size = max(1, int(self.pop_size * self.best_sel_percent))
                parent = random.choice(self.population[:pool_size])
                
            elif self.selection_method == "Turniejowa":
                tournament = random.sample(self.population, min(self.tournament_size, self.pop_size))
                reverse_sort = True if self.opt_type == "Max" else False
                tournament.sort(key=lambda x: x.objective_val, reverse=reverse_sort)
                parent = tournament[0]
                
            elif self.selection_method == "Ruletki":
                objs = [ind.objective_val for ind in self.population]
                if self.opt_type == "Min":
                    max_obj = max(objs)
                    fitnesses = [max_obj - obj + 1e-6 for obj in objs] # avoid 0
                else:
                    min_obj = min(objs)
                    fitnesses = [obj - min_obj + 1e-6 for obj in objs]
                    
                total_fit = sum(fitnesses)
                r = random.uniform(0, total_fit)
                curr = 0.0
                parent = self.population[-1]
                for i, fit in enumerate(fitnesses):
                    curr += fit
                    if curr >= r:
                        parent = self.population[i]
                        break
            else:
                parent = random.choice(self.population)
                
            new_pop.append(copy.deepcopy(parent))
            
        return new_pop

    def crossover(self):
        new_pop = []
        # Carry over elite untouched
        for i in range(self.elitism_count):
            if i < len(self.population):
                new_pop.append(self.population[i])
        
        # Remaining population undergoes crossover
        remaining = self.population[self.elitism_count:]
        random.shuffle(remaining)
        
        for i in range(0, len(remaining), 2):
            if i + 1 < len(remaining):
                p1 = remaining[i]
                p2 = remaining[i+1]
                if random.random() < self.prob_cross:
                    c1, c2 = self._crossover_individuals(p1.chromosome, p2.chromosome)
                    new_pop.append(Individual(c1))
                    new_pop.append(Individual(c2))
                else:
                    new_pop.append(Individual(p1.chromosome[:]))
                    new_pop.append(Individual(p2.chromosome[:]))
            else:
                # If odd remaining, just carry it over
                new_pop.append(Individual(remaining[i].chromosome[:]))
                
        self.population = new_pop

    def _crossover_individuals(self, chrom1, chrom2):
        l = self.total_bits
        c1 = chrom1[:]
        c2 = chrom2[:]
        
        if self.crossover_method == "Jednopunktowe":
            pt = random.randint(1, l - 1) if l > 1 else 0
            c1 = chrom1[:pt] + chrom2[pt:]
            c2 = chrom2[:pt] + chrom1[pt:]
            
        elif self.crossover_method == "Dwupunktowe":
            if l > 2:
                pt1 = random.randint(1, l - 2)
                pt2 = random.randint(pt1 + 1, l - 1)
                c1 = chrom1[:pt1] + chrom2[pt1:pt2] + chrom1[pt2:]
                c2 = chrom2[:pt1] + chrom1[pt1:pt2] + chrom2[pt2:]
            
        elif self.crossover_method in ["Jednorodne", "Ziarniste"]:
            for i in range(l):
                if random.random() < 0.5:
                    c1[i], c2[i] = c2[i], c1[i]
                    
        return c1, c2

    def mutation(self):
        for i in range(self.elitism_count, len(self.population)):
            if random.random() < self.prob_mut:
                self._mutate_individual(self.population[i])

    def _mutate_individual(self, ind):
        chrom = ind.chromosome
        l = self.total_bits
        
        if self.mutation_method == "Jednopunktowa":
            pt = random.randint(0, l - 1) if l > 0 else 0
            chrom[pt] = 1 - chrom[pt]
            
        elif self.mutation_method == "Dwupunktowa":
            if l > 1:
                pt1 = random.randint(0, l - 1)
                pt2 = random.randint(0, l - 1)
                chrom[pt1] = 1 - chrom[pt1]
                if pt1 != pt2:
                    chrom[pt2] = 1 - chrom[pt2]
                
        elif self.mutation_method == "Brzegowa":
             # set a random variable's segment to all 0s or all 1s
             if self.num_vars > 0 and self.bits_per_var > 0:
                 var_idx = random.randint(0, self.num_vars - 1)
                 val = 1 if random.random() < 0.5 else 0
                 for j in range(var_idx * self.bits_per_var, (var_idx + 1) * self.bits_per_var):
                     chrom[j] = val

    def inversion(self):
        for i in range(self.elitism_count, len(self.population)):
            if random.random() < self.prob_inv:
                chrom = self.population[i].chromosome
                l = self.total_bits
                if l > 1:
                    pt1 = random.randint(0, l - 1)
                    pt2 = random.randint(0, l - 1)
                    if pt1 > pt2:
                        pt1, pt2 = pt2, pt1
                    if pt1 < pt2:
                        sub = chrom[pt1:pt2+1]
                        sub.reverse()
                        chrom[pt1:pt2+1] = sub

    def run(self):
        self.initialize_population()
        self.evaluate_population()
        
        for epoch in range(self.epochs):
            # Record stats
            objs = [ind.objective_val for ind in self.population]
            self.best_history.append(self.population[0].objective_val)
            self.avg_history.append(sum(objs)/len(objs))
            self.worst_history.append(self.population[-1].objective_val)
            
            # Evolution
            self.population = self.select()
            self.crossover()
            self.mutation()
            self.inversion()
            self.evaluate_population()

        # Record final
        objs = [ind.objective_val for ind in self.population]
        self.best_history.append(self.population[0].objective_val)
        self.avg_history.append(sum(objs)/len(objs))
        self.worst_history.append(self.population[-1].objective_val)
