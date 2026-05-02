import copy
import math
import random


class Individual:
    """Chromosom rzeczywisty: lista liczb rzeczywistych (po jednym genie na zmienną)."""

    def __init__(self, chromosome):
        self.chromosome = chromosome  # list[float]
        self.real_values = []
        self.fitness = 0.0
        self.objective_val = 0.0


class RealGeneticAlgorithm:
    """GA z kodowaniem rzeczywistym — krzyżowania arytmetyczne, liniowe, BLX-α, α-β, uśredniające;
    mutacja równomierna i Gaussa."""

    def __init__(
        self,
        fitness_func,
        num_vars,
        domain,
        pop_size,
        epochs,
        prob_cross,
        prob_mut,
        prob_inv,
        elitism_count,
        selection_method="Turniejowa",
        crossover_method="Arytmetyczne",
        mutation_method="Równomierna",
        opt_type="Min",
        tournament_size=3,
        best_sel_percent=0.2,
        *,
        blx_alpha=0.5,
        ab_alpha=0.5,
        ab_beta=0.5,
        gaussian_sigma_frac=0.1,
    ):
        self.fitness_func = fitness_func
        self.num_vars = num_vars
        self.domain = domain
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
        self.blx_alpha = blx_alpha
        self.ab_alpha = ab_alpha
        self.ab_beta = ab_beta
        self.gaussian_sigma_frac = gaussian_sigma_frac

        self.population = []
        self.best_history = []
        self.avg_history = []
        self.worst_history = []
        self.best_solution_ever = None

    def _clip(self, values):
        a, b = self.domain
        return [max(a, min(b, float(x))) for x in values]

    def initialize_population(self):
        a, b = self.domain
        self.population = []
        for _ in range(self.pop_size):
            chrom = [random.uniform(a, b) for _ in range(self.num_vars)]
            ind = Individual(chrom)
            self.population.append(ind)

    def evaluate_population(self):
        for ind in self.population:
            ind.chromosome = self._clip(ind.chromosome)
            ind.real_values = ind.chromosome[:]
            ind.objective_val = self.fitness_func(ind.real_values)
            ind.fitness = ind.objective_val

        reverse_sort = self.opt_type == "Max"
        self.population.sort(key=lambda x: x.objective_val, reverse=reverse_sort)

        if self.best_solution_ever is None:
            self.best_solution_ever = copy.deepcopy(self.population[0])
        else:
            if self.opt_type == "Min" and self.population[0].objective_val < self.best_solution_ever.objective_val:
                self.best_solution_ever = copy.deepcopy(self.population[0])
            elif self.opt_type == "Max" and self.population[0].objective_val > self.best_solution_ever.objective_val:
                self.best_solution_ever = copy.deepcopy(self.population[0])

    def select(self):
        new_pop = []
        if self.elitism_count > 0:
            for i in range(min(self.elitism_count, self.pop_size)):
                new_pop.append(copy.deepcopy(self.population[i]))

        while len(new_pop) < self.pop_size:
            if self.selection_method == "Najlepszych":
                pool_size = max(1, int(self.pop_size * self.best_sel_percent))
                parent = random.choice(self.population[:pool_size])

            elif self.selection_method == "Turniejowa":
                tournament = random.sample(self.population, min(self.tournament_size, self.pop_size))
                reverse_sort = self.opt_type == "Max"
                tournament.sort(key=lambda x: x.objective_val, reverse=reverse_sort)
                parent = tournament[0]

            elif self.selection_method == "Ruletki":
                objs = [ind.objective_val for ind in self.population]
                if self.opt_type == "Min":
                    max_obj = max(objs)
                    fitnesses = [max_obj - obj + 1e-6 for obj in objs]
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
        for i in range(self.elitism_count):
            if i < len(self.population):
                new_pop.append(copy.deepcopy(self.population[i]))

        remaining = self.population[self.elitism_count :]
        random.shuffle(remaining)

        for i in range(0, len(remaining), 2):
            if i + 1 < len(remaining):
                p1 = remaining[i].chromosome
                p2 = remaining[i + 1].chromosome
                if random.random() < self.prob_cross:
                    c1, c2 = self._crossover_vectors(p1, p2)
                    new_pop.append(Individual(self._clip(c1)))
                    new_pop.append(Individual(self._clip(c2)))
                else:
                    new_pop.append(Individual(p1[:]))
                    new_pop.append(Individual(p2[:]))
            else:
                new_pop.append(Individual(remaining[i].chromosome[:]))

        self.population = new_pop

    def _crossover_vectors(self, v1, v2):
        m = self.crossover_method

        if m == "Arytmetyczne":
            lam = random.random()
            c1 = [lam * a + (1 - lam) * b for a, b in zip(v1, v2)]
            c2 = [(1 - lam) * a + lam * b for a, b in zip(v1, v2)]
            return c1, c2

        if m == "Liniowe":
            o1 = [(a + b) * 0.5 for a, b in zip(v1, v2)]
            o2 = [1.5 * a - 0.5 * b for a, b in zip(v1, v2)]
            o3 = [-0.5 * a + 1.5 * b for a, b in zip(v1, v2)]
            cands = [self._clip(o) for o in (o1, o2, o3)]
            i, j = random.sample(range(len(cands)), k=2)
            return cands[i][:], cands[j][:]

        if m == "Mieszające alfa":
            alpha = self.blx_alpha
            c1, c2 = [], []
            for a, b in zip(v1, v2):
                lo, hi = min(a, b), max(a, b)
                d = hi - lo
                ext = alpha * d if d > 0 else alpha * abs(self.domain[1] - self.domain[0]) * 1e-6
                c1.append(random.uniform(lo - ext, hi + ext))
                c2.append(random.uniform(lo - ext, hi + ext))
            return c1, c2

        if m == "Alfa-beta":
            alp, bet = self.ab_alpha, self.ab_beta
            c1, c2 = [], []
            for a, b in zip(v1, v2):
                lo, hi = min(a, b), max(a, b)
                d = hi - lo
                if d <= 1e-15:
                    d = abs(self.domain[1] - self.domain[0]) * 1e-12
                c1.append(random.uniform(lo - alp * d, hi + bet * d))
                c2.append(random.uniform(lo - alp * d, hi + bet * d))
            return c1, c2

        if m == "Uśredniające":
            avg = [(a + b) * 0.5 for a, b in zip(v1, v2)]
            return avg[:], avg[:]

        # Domyślnie jak arytmetyczne
        lam = random.random()
        c1 = [lam * a + (1 - lam) * b for a, b in zip(v1, v2)]
        c2 = [(1 - lam) * a + lam * b for a, b in zip(v1, v2)]
        return c1, c2

    def mutation(self):
        for i in range(self.elitism_count, len(self.population)):
            if random.random() < self.prob_mut:
                self._mutate_individual(self.population[i])

    def _mutate_individual(self, ind):
        chrom = ind.chromosome
        a, b = self.domain
        n = len(chrom)
        sigma = self.gaussian_sigma_frac * abs(b - a)
        sigma = max(sigma, 1e-12)

        if self.mutation_method == "Równomierna":
            j = random.randrange(n)
            chrom[j] = random.uniform(a, b)

        elif self.mutation_method == "Gaussa":
            j = random.randrange(n)
            chrom[j] = chrom[j] + random.gauss(0, sigma)
            chrom[j] = max(a, min(b, chrom[j]))

    def inversion(self):
        for i in range(self.elitism_count, len(self.population)):
            if random.random() < self.prob_inv:
                chrom = self.population[i].chromosome
                n = len(chrom)
                if n > 1:
                    p1 = random.randint(0, n - 1)
                    p2 = random.randint(0, n - 1)
                    if p1 > p2:
                        p1, p2 = p2, p1
                    if p1 < p2:
                        sub = chrom[p1 : p2 + 1]
                        sub.reverse()
                        chrom[p1 : p2 + 1] = sub

    def run(self):
        self.initialize_population()
        self.evaluate_population()

        for _ in range(self.epochs):
            objs = [ind.objective_val for ind in self.population]
            self.best_history.append(self.population[0].objective_val)
            self.avg_history.append(sum(objs) / len(objs))
            self.worst_history.append(self.population[-1].objective_val)

            self.population = self.select()
            self.crossover()
            self.mutation()
            self.inversion()
            self.evaluate_population()

        objs = [ind.objective_val for ind in self.population]
        self.best_history.append(self.population[0].objective_val)
        self.avg_history.append(sum(objs) / len(objs))
        self.worst_history.append(self.population[-1].objective_val)


class GeneticAlgorithm:
    """GA z kodowaniem binarnym (Projekt 1) — do porównań w sprawozdaniu P1 vs P2."""

    def __init__(
        self,
        fitness_func,
        num_vars,
        domain,
        precision_decimals,
        pop_size,
        epochs,
        prob_cross,
        prob_mut,
        prob_inv,
        elitism_count,
        selection_method="Turniejowa",
        crossover_method="Jednopunktowe",
        mutation_method="Jednopunktowa",
        opt_type="Min",
        tournament_size=3,
        best_sel_percent=0.2,
    ):
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

        self.best_history = []
        self.avg_history = []
        self.worst_history = []
        self.best_solution_ever = None

    def _calculate_bits(self):
        a, b = self.domain
        diff = b - a
        points = diff * (10 ** self.precision_decimals) + 1
        return max(1, math.ceil(math.log2(max(points, 2))))

    def _decode(self, chromosome):
        real_vals = []
        a, b = self.domain
        max_dec = (2 ** self.bits_per_var) - 1

        if max_dec == 0:
            return [a] * self.num_vars

        for i in range(self.num_vars):
            sub_chrom = chromosome[i * self.bits_per_var : (i + 1) * self.bits_per_var]
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
            ind.fitness = ind.objective_val

        reverse_sort = self.opt_type == "Max"
        self.population.sort(key=lambda x: x.objective_val, reverse=reverse_sort)

        if self.best_solution_ever is None:
            self.best_solution_ever = copy.deepcopy(self.population[0])
        else:
            if self.opt_type == "Min" and self.population[0].objective_val < self.best_solution_ever.objective_val:
                self.best_solution_ever = copy.deepcopy(self.population[0])
            elif self.opt_type == "Max" and self.population[0].objective_val > self.best_solution_ever.objective_val:
                self.best_solution_ever = copy.deepcopy(self.population[0])

    def select(self):
        new_pop = []
        if self.elitism_count > 0:
            for i in range(min(self.elitism_count, self.pop_size)):
                new_pop.append(copy.deepcopy(self.population[i]))

        while len(new_pop) < self.pop_size:
            if self.selection_method == "Najlepszych":
                pool_size = max(1, int(self.pop_size * self.best_sel_percent))
                parent = random.choice(self.population[:pool_size])

            elif self.selection_method == "Turniejowa":
                tournament = random.sample(self.population, min(self.tournament_size, self.pop_size))
                reverse_sort = self.opt_type == "Max"
                tournament.sort(key=lambda x: x.objective_val, reverse=reverse_sort)
                parent = tournament[0]

            elif self.selection_method == "Ruletki":
                objs = [ind.objective_val for ind in self.population]
                if self.opt_type == "Min":
                    max_obj = max(objs)
                    fitnesses = [max_obj - obj + 1e-6 for obj in objs]
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
        for i in range(self.elitism_count):
            if i < len(self.population):
                new_pop.append(copy.deepcopy(self.population[i]))

        remaining = self.population[self.elitism_count :]
        random.shuffle(remaining)

        for i in range(0, len(remaining), 2):
            if i + 1 < len(remaining):
                p1 = remaining[i]
                p2 = remaining[i + 1]
                if random.random() < self.prob_cross:
                    c1, c2 = self._crossover_individuals(p1.chromosome, p2.chromosome)
                    new_pop.append(Individual(c1))
                    new_pop.append(Individual(c2))
                else:
                    new_pop.append(Individual(p1.chromosome[:]))
                    new_pop.append(Individual(p2.chromosome[:]))
            else:
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

        elif self.crossover_method == "Jednorodne":
            for i in range(l):
                if random.random() < 0.5:
                    c1[i], c2[i] = c2[i], c1[i]

        elif self.crossover_method == "Ziarniste":
            for i in range(self.num_vars):
                if random.random() < 0.5:
                    start_idx = i * self.bits_per_var
                    end_idx = start_idx + self.bits_per_var
                    c1[start_idx:end_idx], c2[start_idx:end_idx] = (
                        c2[start_idx:end_idx],
                        c1[start_idx:end_idx],
                    )

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
                        sub = chrom[pt1 : pt2 + 1]
                        sub.reverse()
                        chrom[pt1 : pt2 + 1] = sub

    def run(self):
        self.initialize_population()
        self.evaluate_population()

        for _ in range(self.epochs):
            objs = [ind.objective_val for ind in self.population]
            self.best_history.append(self.population[0].objective_val)
            self.avg_history.append(sum(objs) / len(objs))
            self.worst_history.append(self.population[-1].objective_val)

            self.population = self.select()
            self.crossover()
            self.mutation()
            self.inversion()
            self.evaluate_population()

        objs = [ind.objective_val for ind in self.population]
        self.best_history.append(self.population[0].objective_val)
        self.avg_history.append(sum(objs) / len(objs))
        self.worst_history.append(self.population[-1].objective_val)
