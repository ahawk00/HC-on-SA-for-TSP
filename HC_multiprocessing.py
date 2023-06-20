import math
import random
import numpy as np
import time
import csv
import matplotlib
import re
import os

matplotlib.use('Agg')  # Use the Agg backend so we don't invoke graphics from children thread (OSX no like that :o)
import matplotlib.pyplot as plt
from multiprocessing import Process  # We use subprocesses instead of threads to leverage that multicore

MODULO = 20


class Coordinates:
    def __init__(self, index, x, y):
        ''' Constructs a new Coordinate object with index, x and y '''
        self.idx = int(index)
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return '{} {} {}'.format(self.idx, self.x, self.y)

    def __repr__(self):
        return str(self)


class TSP:
    '''
    Definition of a TSP. Provides methods for storing and shuffling a route. Initiated by file.
    '''

    def __init__(self, file_name):
        self.coords = []

        # extracts the coordinates from inputted file
        file = open(file_name).read().split('\n')
        i = 0
        for line in file:
            i += 1

            if line == 'NODE_COORD_SECTION':
                for element in file[i:]:
                    if element == 'EOF':
                        break
                    c = element.split()
                    self.add_coord(Coordinates(c[0], c[1], c[2]))

    def __str__(self):
        return '{}'.format(self.coords)

    def add_coord(self, coord):
        self.coords.append(coord)

    def get_coord(self, index):
        '''Get coordinate with index. Returns none if not found'''
        for coord in self.coords:
            if coord.idx == index:
                return coord
        return None

    def initial_route(self):
        ''' Shuffle the route of the problem '''
        random.shuffle(self.coords)
        return self.coords


class SA:
    def __init__(self, route):
        self.initial_route = TSP(route).initial_route()
        self.init_temp = 2000
        self.end_temp = 0.00001
        self.temp = 1000
        self.alpha = 0.9
        self.step = 1
        self.max_step = 100000

        self.best_dist = 0
        self.best_route = self.initial_route

        self.curr_dist = 0
        self.curr_route = self.initial_route

        self.fitness_list = list()
        self.temp_list = list()

        self.start_time = time.time()

    def calc_eucl_dist(self, node1, node2):
        '''Calculate euclidean distance between node1 and node2'''
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def calc_tot_dist(self, route):  # need variable added to input the route required to calculate
        ''' Calculates total route distance'''
        distance = 0
        prev_coord = None
        for coord in route:
            if prev_coord is not None:
                distance += self.calc_eucl_dist(prev_coord, coord)
            prev_coord = coord

        # distance from final to start destination
        distance += self.calc_eucl_dist(prev_coord, route[0])

        return distance

    def inverse_operator(self, i, j):
        '''Applies inverse mutation method to current route to create a new candidate route'''
        candidate = [i for i in self.curr_route]
        candidate[j: (j + i)] = reversed(candidate[j: (j + i)])

        return candidate

    # def insert_operator(self, i, j):
    #     '''Applies insert mutation method to current route to create a new candidate route'''
    #     candidate = [i for i in self.curr_route]
    #     candidate.insert(j, candidate[i])
    #     if j < i:
    #         i += 1
    #     candidate.pop(i)
    #
    #     return candidate
    #
    # def swap_operator(self, i, j):
    #     '''Applies swap mutation method to current route to create a new candidate route'''
    #     candidate = [i for i in self.curr_route]
    #     candidate[i], candidate[j] = candidate[j], candidate[i]
    #
    #     return candidate

    def new_candidate(self):
        '''
        Mutates route current route.
        '''
        i = random.randint(2, len(self.initial_route) - 1)
        j = random.randint(0, i)

        candidate = [i for i in self.curr_route]
        candidate[j: (j + i)] = reversed(candidate[j: (j + i)])

        if (j + i) >= len(candidate):
            current_dist_j = self.calc_eucl_dist(self.curr_route[j - 1], self.curr_route[j])
            new_dist_j = self.calc_eucl_dist(candidate[j - 1], candidate[j])

            current_dist_i = self.calc_eucl_dist(self.curr_route[-1], self.curr_route[0])
            new_dist_i = self.calc_eucl_dist(candidate[-1], candidate[0])

            candidate_dist = (self.curr_dist - (current_dist_i + current_dist_j)) + (new_dist_i + new_dist_j)

            return candidate, candidate_dist

        else:
            if j == 0:
                current_dist_j = self.calc_eucl_dist(self.curr_route[-1], self.curr_route[j])
                new_dist_j = self.calc_eucl_dist(candidate[-1], candidate[j])
            else:
                current_dist_j = self.calc_eucl_dist(self.curr_route[j - 1], self.curr_route[j])
                new_dist_j = self.calc_eucl_dist(candidate[j - 1], candidate[j])

            current_dist_i = self.calc_eucl_dist(self.curr_route[i + j - 1], self.curr_route[i + j])
            new_dist_i = self.calc_eucl_dist(candidate[i + j - 1], candidate[i + j])

        # Get the difference of distance
        candidate_dist = round((self.curr_dist - (current_dist_i + current_dist_j)) + (new_dist_i + new_dist_j), 10)

        return candidate, candidate_dist

    def acceptance_prob(self, new_dist, temp):
        '''
        Calculates the acceptance probability.
        Based on the difference of the new and current distance and the current temp
        '''
        return math.exp(-abs(new_dist - self.curr_dist) / temp)

    def accept(self, new_route, new_dist, temp):
        '''
        Determines acceptance of the new candidate.
        Based on temperature and difference of new candidate and current best candidate
        '''
        if new_dist < self.curr_dist:
            self.curr_route, self.curr_dist = new_route, new_dist
            if new_dist < self.best_dist:
                self.best_route, self.best_dist = new_route, new_dist
        else:
            if random.random() < self.acceptance_prob(new_dist, temp):
                self.curr_route, self.curr_dist = new_route, new_dist

    def anneal(self, temp_list):
        '''Annealing steps'''
        # initialize the current and best distance and route
        self.curr_route = self.initial_route
        self.curr_dist = self.calc_tot_dist(self.initial_route)
        self.best_dist = self.curr_dist

        # annealing step to find the best solution
        for temp in temp_list:
            # while self.temp > self.end_temp and self.step < self.max_step:
            new_route, new_dist = self.new_candidate()
            self.accept(new_route, new_dist, temp)

            # temp and iteration already predetermined, iterate through list given by hill climber
            # self.update_temp(cooling_schedule)

            self.fitness_list.append(self.curr_dist)
            self.step += 1

        return self.best_dist


class HC:
    def __init__(self, temp=1000, alpha=0.9, end_temp=0.00001):
        self.init_temp_list = list()
        self.current_temp_list = list()
        self.new_temp_list = list()
        self.best_temp_list = list()

        self.curr_fitness = 0
        self.new_fitness = 0
        self.best_fitness = 0

        self.init_temp = 1000
        self.temp = 1000
        self.end_temp = 0.00001
        self.step = 1
        self.max_SA_step = 100000

        self.alpha = 0.9

        self.start_time = time.time()

        self.fitness_values = list()  # TODO append final distance to this list
        self.returned_temp_list = list()

        self.accepted = 0

    def initialize_temp_list(self, cool_type):
        '''Updates the temperature, depends on selected cooling schedule type'''

        temp_list = []
        while self.step < self.max_SA_step:
            if cool_type == 'linear':  # 500
                self.temp = self.init_temp * (1 - (self.step / self.max_SA_step))
                temp_list.append(self.temp)
            if cool_type == 'sigmoid':  # 500
                self.temp = self.init_temp / (1 + np.exp(10 ** -4 * (self.step - (self.max_SA_step / 2))))
                temp_list.append(self.temp)
            if cool_type == 'geometric':  # 500.64199286968045 TODO doesnt converge to 0
                self.temp = self.temp * 0.9999
                temp_list.append(self.temp)
            if cool_type == 'stairs':  # 500.05
                t_step = 1000 / 10
                self.temp = t_step * (10 - int(self.step / (self.max_SA_step / 10)))
                temp_list.append(self.temp)
            if cool_type == 'cosine':  # 499.95032633273973
                self.temp = self.init_temp / 2 * np.cos(self.step / 1516) + self.init_temp / 2
                temp_list.append(self.temp)
            if cool_type == 'linear_reheat':  # 99.91223731748175
                division = self.max_SA_step / 10
                t_0 = 1000 * 0.5 ** (int(self.step / division))
                self.temp = t_0 - t_0 / division * (self.step - (int(self.step / division) * division))
                temp_list.append(self.temp)
            if cool_type == 'GG_1':
                self.temp = 1 / (np.log(self.step + 1))
                temp_list.append(self.temp)
            if cool_type == 'GG_50':
                self.temp = 50 / (np.log(self.step + 1))
                temp_list.append(self.temp)
            if cool_type == 'GG_195075':
                self.temp = 195075 / (np.log(self.step + 1))
                temp_list.append(self.temp)

            self.init_temp_list = temp_list

            self.step += 1

        return self.init_temp_list

    def new_temp_candidate(self):
        '''Mutates current temperature list to create a new one'''
        candidate = [i for i in self.current_temp_list]

        indices = random.sample(range(len(self.current_temp_list)), 1000)

        for index in indices:
            candidate[index] += random.randrange(-30, 30)
            if candidate[index] <= 0:
                candidate[index] = 1

        return candidate

    def accept(self, new_temp_list, new_fitness):
        '''
        Checks if new temp fitness is better than previous on.
        If yes accept it as current temperature list
        '''
        if new_fitness < self.curr_fitness:
            self.curr_fitness = new_fitness
            self.current_temp_list = new_temp_list
            self.accepted += 1
            if new_fitness < self.best_fitness:
                self.best_temp_list, self.best_fitness = new_temp_list, new_fitness

    def hill_climber(self, file, cooling_type, nr_iterations):
        '''Hill climbing steps'''

        # initializes the temperature list and fitness
        temp_list = self.initialize_temp_list(cooling_type)
        self.current_temp_list = temp_list

        init_problem = SA(file)
        self.curr_fitness = init_problem.anneal(self.current_temp_list)
        self.best_fitness = self.curr_fitness

        # hill climbing on SA to find the best temperature list
        step = 0
        while step <= nr_iterations:
            new_temp_list = self.new_temp_candidate()

            new_problem = SA(file)
            best_dist = new_problem.anneal(new_temp_list)

            self.accept(new_temp_list, best_dist)

            self.fitness_values.append(self.curr_fitness)

            step += 1

        run_time = time.time() - self.start_time

        return self.fitness_values, run_time, self.current_temp_list, self.curr_fitness, self.best_fitness, \
            self.accepted, self.best_temp_list


TSP_problems = ['a280.tsp', 'berlin52.tsp', 'bier127.tsp', 'ch130.tsp', 'ch150.tsp',
                'd198.tsp', 'd493.tsp', 'd657.tsp', 'd1291.tsp', 'eil51.tsp',
                'eil101.tsp', 'fl417.tsp', 'kroA100.tsp', 'kroB100.tsp', 'kroC100.tsp',
                'kroD100.tsp', 'kroE100.tsp', 'kroA150.tsp', 'kroB150.tsp', 'kroA200.tsp',
                'kroB200.tsp', 'lin105.tsp', 'lin318.tsp', 'p654.tsp', 'pcb442.tsp', 'pcb1173.tsp',
                'pr107.tsp', 'pr124.tsp', 'pr136.tsp', 'pr144.tsp',
                'pr152.tsp', 'pr226.tsp', 'pr299.tsp',
                'rat99.tsp', 'rat195.tsp', 'rat575.tsp', 'rat783.tsp', 'rd100.tsp', 'rd400.tsp', 'st70.tsp',
                'rl1304.tsp', 'rl1323.tsp', 'ts225.tsp', 'tsp225.tsp', 'pr76.tsp',
                'u159.tsp', 'u574.tsp', 'u724.tsp', 'u1060.tsp', 'vm1084.tsp']

cooling_schedules = ['linear', 'sigmoid', 'geometric', 'stairs', 'cosine', 'linear_reheat', 'GG_1', 'GG_50',
                      'GG_195075']

def worker(prob, nr_iterations):
    try:
        # read TSP file
        file = 'datasets/' + prob
        prob_name = prob.split('.')

        # create results file

        # directory_path = '/Users/amberhawkins/Desktop/Thesis/Code/HC_' + str(nr_iterations) + '/' + prob_name[0] + '/'
        # directory_path = '/Users/amberhawkins/Desktop/Thesis/Code/' + prob_name[0] + '_linear/'

        directory_path = "/var/scratch/ahs354/HC_" + str(nr_iterations) + '_corrected/' + prob_name[0] + '/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # run hill climber
        # temp_values = []
        for cool in cooling_schedules:
            print("Handling problem %s.%s" % (prob, cool))

            result_file = open(directory_path + cool + "_results.txt", "w")
            result_writer = csv.writer(result_file)
            result_writer.writerow(
                ['Problem', 'Cooling Schedule', 'Iteration Number', 'Run Time', 'Nr of Better Solutions Found',
                 'Best Fitness', 'Last Fitness', 'Best Temperature List'])

            for i in range(10):
                fitness_file = open(directory_path + str(i) + "_" + cool + "_fitness_results.txt", "w")
                fitness_writer = csv.writer(fitness_file)
                fitness_writer.writerow(
                    ['Problem', 'Cooling Schedule', 'Iteration Number', 'Fitness', 'Tour'])

                problem = HC(temp=1000, alpha=0.9, end_temp=0.00001)
                fitness_value, run_time_HC, temp_list, last_fitness, best_fitness, accepted, best_temp_list, \
                    = problem.hill_climber(file=file, cooling_type=cool, nr_iterations=nr_iterations)
                print('{} - {}, {}, {} - {}, {}'.format(i, prob, cool, run_time_HC, fitness_value, best_fitness))

                result_writer.writerow([prob_name[0], i, cool, nr_iterations, run_time_HC,
                                        accepted, best_fitness, last_fitness, best_temp_list])

                fitness = problem.fitness_values

                iteration = 1
                for fit in fitness:
                    if iteration % MODULO == 0:
                        fitness_writer.writerow([prob_name[0], cool, iteration, fit, temp_list])
                    else:
                        fitness_writer.writerow([prob_name[0], cool, iteration, fit, None])
                    iteration += 1

    except Exception as e:
        print("FAIL! Something went wrong with prob %s : %s" % (prob, e))
    finally:
        # jobs.task_done() # Declare job is done
        print("Finished handling problem {}".format(prob))


if __name__ == '__main__':

    # create new directory to save plots and results
    nr_iterations = 100

    for prob in TSP_problems:
        p = Process(target=worker, args=(prob, nr_iterations))
        p.start()
