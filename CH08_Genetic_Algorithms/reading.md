# Genetic Algorithms

- Understanding evolutionary and genetic algorithms
- Fundamental concepts in genetic algorithms
- Generating a bit pattern with predefined parameters
- Visualizing the progress of the evolution
- Solving the symbol regression problem
- Building an intelligent robot controller

## Table of Contents

- [Genetic Algorithms](#genetic-algorithms)
  - [Table of Contents](#table-of-contents)
  - [evolutionary and genetic algorithms](#evolutionary-and-genetic-algorithms)
  - [Fundamental concepts in genetic algorithms](#fundamental-concepts-in-genetic-algorithms)
  - [Examples](#examples)
  - [Visualizing the evolution](#visualizing-the-evolution)
  - [Solving the symbol regression problem](#solving-the-symbol-regression-problem)
  - [Building an intelligent robot controller](#building-an-intelligent-robot-controller)

## evolutionary and genetic algorithms

- Optimization techniques inspired by the process of natural evolution. 
- They are used to find solutions to complex optimization and search problems. 
- Both approaches work by evolving a population of potential solutions over multiple generations to find the best solution.

**Genetic Algorithms (GAs):**
1. **Inspired by Genetics:** Genetic algorithms are inspired by the principles of genetics and natural selection.
2. **Encoding Solutions:** Solutions to a problem are encoded as individuals in a population, often as strings of binary or real-valued numbers.
3. **Selection:** Individuals are selected from the population based on their fitness, which represents how well they solve the problem.
4. **Crossover (Recombination):** Pairs of individuals are combined to produce offspring with traits inherited from both parents.
5. **Mutation:** Random changes are applied to some of the offspring to introduce genetic diversity.
6. **Termination:** The process continues for a fixed number of generations or until a convergence criterion is met.
7. **Applications:** Genetic algorithms are used in various fields, including optimization, machine learning, game playing, and engineering design.

**Evolutionary Algorithms (EAs):**
1. **Generalization:** Evolutionary algorithms are a broader class of algorithms that encompass genetic algorithms and other evolutionary strategies.
2. **Encoding Solutions:** Solutions can be encoded in various ways, including binary strings, real-valued vectors, or more complex data structures.
3. **Selection:** Individuals are selected based on fitness, similar to genetic algorithms.
4. **Recombination and Mutation:** Evolutionary algorithms may use various recombination and mutation strategies depending on the problem.
5. **Termination:** Like genetic algorithms, the process continues for a fixed number of generations or until convergence.
6. **Applications:** Evolutionary algorithms are used in optimization problems, neural network training, robotics, and more.

In both genetic and evolutionary algorithms, the key idea is to explore the search space of possible solutions systematically. These algorithms are particularly useful for complex optimization problems where traditional search techniques may struggle.

Genetic algorithms are a specific subset of evolutionary algorithms that adhere to a particular set of encoding, selection, crossover, and mutation strategies. Evolutionary algorithms provide more flexibility and can be adapted to a wider range of problem domains.

## Fundamental concepts in genetic algorithms

Genetic algorithms (GAs) are optimization algorithms inspired by the process of natural selection. They evolve a population of potential solutions to a problem over multiple generations. Here are some fundamental concepts in genetic algorithms, along with example Python code:

**1. Chromosome Representation:** In genetic algorithms, potential solutions are encoded as chromosomes, often represented as strings of binary digits or real-valued vectors.

**Example Code:**

```python
# Chromosome representation as a binary string
chromosome = '1101001010110010'

# Chromosome representation as a real-valued vector
chromosome = [0.5, 0.2, 0.8, 0.1]
```

**2. Population:** A population consists of a group of individuals (chromosomes), each representing a potential solution to the problem.

**Example Code:**

```python
population = ['1010101010101010', '1100110011001100', '0011001100110011']
```

**3. Fitness Function:** A fitness function evaluates how well a chromosome solves the problem. It assigns a fitness score to each chromosome in the population.

**Example Code:**

```python
def fitness_function(chromosome):
    # Example fitness function (maximizing)
    return sum(int(bit) for bit in chromosome)

# Calculate fitness for a chromosome
chromosome = '1101001010110010'
fitness = fitness_function(chromosome)
```

**4. Selection:** Selection involves choosing individuals from the population to become parents for the next generation. Fitter individuals are more likely to be selected.

**Example Code:**

```python
import random

def selection(population, fitness_scores):
    # Example roulette wheel selection
    selected = []
    total_fitness = sum(fitness_scores)
    for _ in range(len(population)):
        rand_value = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        for i, fitness in enumerate(fitness_scores):
            cumulative_fitness += fitness
            if cumulative_fitness >= rand_value:
                selected.append(population[i])
                break
    return selected
```

**5. Crossover (Recombination):** Crossover combines genetic material from two parents to create one or more offspring. Various crossover techniques can be used.

**Example Code:**

```python
def crossover(parent1, parent2):
    # Example one-point crossover
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
```

**6. Mutation:** Mutation introduces small random changes into the offspring's chromosomes to maintain genetic diversity.

**Example Code:**

```python
def mutation(chromosome, mutation_rate):
    # Example bit-flip mutation
    mutated_chromosome = ''
    for bit in chromosome:
        if random.random() < mutation_rate:
            mutated_chromosome += '0' if bit == '1' else '1'
        else:
            mutated_chromosome += bit
    return mutated_chromosome
```

**7. Termination:** Termination criteria determine when to stop the genetic algorithm. Common criteria include reaching a maximum number of generations or finding a satisfactory solution.

**Example Code:**

```python
def termination_condition(generation, max_generations):
    # Example termination condition (reaching a maximum number of generations)
    return generation >= max_generations
```

**Example Genetic Algorithm:**

Here's an example of a simple genetic algorithm that maximizes a fitness function:

```python
import random

# Genetic algorithm parameters
population_size = 50
chromosome_length = 16
mutation_rate = 0.01
max_generations = 100

# Initialize population
population = [''.join(random.choice('01') for _ in range(chromosome_length)) for _ in range(population_size)]

# Main loop
for generation in range(max_generations):
    # Evaluate fitness for each chromosome
    fitness_scores = [fitness_function(chromosome) for chromosome in population]
    
    # Select parents
    selected_parents = selection(population, fitness_scores)
    
    # Create offspring through crossover and mutation
    offspring = []
    while len(offspring) < population_size:
        parent1, parent2 = random.sample(selected_parents, 2)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        offspring.extend([child1, child2])
    
    # Replace old population with offspring
    population = offspring
    
    # Check termination condition
    if termination_condition(generation, max_generations):
        break

# Find the best solution
best_solution = max(population, key=fitness_function)
best_fitness = fitness_function(best_solution)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
```

## Examples

The One Max problem is a simple binary optimization problem where the goal is to find a binary string (chromosome) of a fixed length that contains as many '1's as possible. It's a classic problem used to introduce genetic algorithms. To solve the One Max problem using Python, you can use the DEAP (Distributed Evolutionary Algorithms in Python) library, which is a powerful framework for evolutionary algorithm development. Here's an example implementation:

First, you'll need to install the DEAP library if you haven't already:

```bash
pip install deap
```

Now, you can implement the One Max problem using DEAP:

```python
import random
from deap import base, creator, tools, algorithms

# Create a fitness maximizing class
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Create an individual with a binary genotype
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the toolbox
toolbox = base.Toolbox()

# Attribute generator: create a random binary digit (0 or 1)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers: create an individual composed of 'attr_bool' elements
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=10)

# Create a population of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation function for the One Max problem
def evaluate(individual):
    return sum(individual),

toolbox.register("evaluate", evaluate)

# Define the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    # Create a population of 100 individuals
    population = toolbox.population(n=100)

    # Number of generations
    ngen = 50

    # Probability of mating two individuals
    cxpb = 0.7

    # Probability of mutating an individual
    mutpb = 0.2

    # Perform the evolutionary algorithm
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)

    # Find the best individual in the final population
    best_individual = tools.selBest(population, k=1)[0]
    print("Best Individual:", best_individual)
    print("Fitness:", best_individual.fitness.values[0])
```

## Visualizing the evolution
## Solving the symbol regression problem
## Building an intelligent robot controller