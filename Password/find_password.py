#Find password suing Genetic Algorithm
import random
import math

secret_password = [2, 0, 1, 8, 0, 4, 2, 4]
password_size = len(secret_password)

def genetic_algorithm(population_size) :

    #Initial Population
    population = []
    step = 0
    for i in range(population_size) :
        #Generate random password
        random_password = generate_random_password(password_size)
        score = fitness(random_password)
        population.append((score, random_password))

    #Generate new generation until criterion met
    best = sorted(population, key = lambda x: x[0], reverse=True)
    while(best[0] != password_size) :

        new_population = []
        while len(new_population) < population_size :

            #select parents
            parent1 = selection(population, 4)
            parent2 = selection(population, 4)

            #Crossover
            offspring1, offspring2 = crossover(parent1, parent2)
            
            #Mutation
            offspring1 = mutation(offspring1, 0.2)
            offspring2 = mutation(offspring2, 0.2)

            new_population.append(offspring1)
            new_population.append(offspring2)

        #Generate new population
        population = new_population
        best = sorted(population, key = lambda x: x[0], reverse=True)[0]
        print(best)
        step+=1
    
    final_password = ""
    for digit in best[1] :
        final_password += str(digit)

    print("Secret Password: " + final_password)
    print("Number of Generation: " + str(step))

def generate_random_password(password_size) :

    password = []
    for _ in range(password_size) :
        digit = random.randint(0, 9)
        password.append(digit)

    return password

def fitness(password) :

    #Fitness score: number of correct digits
    score = 0
    for i in range(password_size) :
        if password[i] == secret_password[i] :
            score += 1

    return score

def selection(population, sample_size) :

    tournament_pool = random.sample(population, sample_size)

    return sorted(tournament_pool, key = lambda x:x[0], reverse=True)[0]

def crossover(parent1, parent2) :

    cut_index = random.randint(0, password_size-1)
    
    parent_1_left = parent1[1][:cut_index]
    parent_1_right = parent1[1][cut_index:]
    parent_2_left = parent2[1][:cut_index]
    parent_2_right = parent2[1][cut_index:]

    offspring1 = (0, parent_1_left + parent_2_right)
    offspring2 = (0, parent_2_left + parent_1_right)

    return offspring1, offspring2

def mutation(offspring, mutation_rate) :

    for i in range(password_size) :

        if random.random() < mutation_rate :
            offspring[1][i] = random.randint(0, 9)

    score = fitness(offspring[1])

    return (score, offspring[1])

if __name__ == "__main__" :

    genetic_algorithm(20)