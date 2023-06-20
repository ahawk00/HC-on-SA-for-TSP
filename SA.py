import math
import random
import numpy as np
import time
import csv
import matplotlib

import traceback
import sys

matplotlib.use('Agg')  # Use the Agg backend so we don't invoke graphics from children thread (OSX no like that :o)
import matplotlib.pyplot as plt
import os
from multiprocessing import Process  # We use subprocesses instead of threads to leverage that multicore
from decimal import *


class Coordinates:
    def __init__(self, index, x, y):
        ''' Constructs a new Coordinate object with index, x and y '''
        self.idx = int(index)
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return '{}'.format(self.idx)

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

            if line == 'NODE_COORD_SECTION' or line == 'DISPLAY_DATA_SECTION':
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
    def __init__(self, route, temp=1000, alpha=0.9, end_temp=0.00001, max_step=100000):
        self.initial_route = TSP(route).initial_route()
        self.init_temp = temp
        self.end_temp = end_temp
        self.temp = temp
        self.alpha = alpha
        self.step = 1
        self.max_step = max_step

        self.best_dist = 0
        self.best_route = self.initial_route

        self.curr_dist = 0
        self.curr_route = self.initial_route

        self.fitness_list = list()

        self.start_time = time.time()

    def calc_eucl_dist(self, node1, node2):
        '''Calculate euclidean distance between node1 and node2'''
        return round(math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2), 10)

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

    def new_candidate(self):
        '''
        Mutates route current route
        Returns the new candidate and distance
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

    def acceptance_prob(self, new_dist):
        '''
        Calculates the acceptance probability.
        Based on the difference of the new and current distance and the current temp
        '''
        return math.exp(-abs(new_dist - self.curr_dist) / self.temp)

    def accept(self, new_route, new_dist):
        '''
        Determines acceptance of the new candidate.
        Based on temperature and difference of new candidate and current best candidate
        '''
        if new_dist < self.curr_dist:
            self.curr_route, self.curr_dist = new_route, new_dist
            if new_dist < self.best_dist:
                self.best_route, self.best_dist = self.curr_route, self.curr_dist
        else:
            if random.random() < self.acceptance_prob(new_dist):
                self.curr_route, self.curr_dist = new_route, new_dist

    # TODO double check that all the cooling schedules work as they should
    def update_temp(self, cool_type):
        '''Updates the temperature, depends on selected cooling schedule type'''
        if cool_type == 'linear':  # 500
            self.temp = self.init_temp * (1 - (self.step / self.max_step))
        if cool_type == 'sigmoid':  # 500
            self.temp = self.init_temp / (1 + np.exp(10 ** -4 * (self.step - (self.max_step / 2))))
        if cool_type == 'geometric':
            self.temp = self.temp * 0.9999
        if cool_type == 'stairs':  # 500.05
            t_step = 1000 / 10
            self.temp = t_step * (10 - int(self.step / (self.max_step / 10)))
        if cool_type == 'cosine':  # 499.95032633273973
            self.temp = self.init_temp / 2 * np.cos(self.step / 1516) + self.init_temp / 2
        if cool_type == 'linear_reheat':
            division = self.max_step / 10
            t_0 = 1000 * 0.5 ** (int(self.step / division))
            self.temp = t_0 - t_0 / division * (self.step - (int(self.step / division) * division))
        if cool_type == 'GG_1':
            self.temp = 1 / (np.log(self.step + 1))
        if cool_type == 'GG_50':
            self.temp = 50 / (np.log(self.step + 1))
        if cool_type == 'GG_195075':
            self.temp = 195075 / (np.log(self.step + 1))

    def anneal(self, cooling_schedule):
        '''Annealing steps'''
        # initialize the current and best distance and route
        initial_dist = self.calc_tot_dist(self.initial_route)
        self.curr_dist = initial_dist
        self.best_dist = initial_dist

        # annealing step to find the best solution
        while self.step <= self.max_step:
            new_route, new_dist = self.new_candidate()
            self.accept(new_route, new_dist)

            # temp and iteration already predetermined, iterate through list given by hill climber
            self.update_temp(cooling_schedule)

            self.step += 1
            self.fitness_list.append(self.curr_dist)

        run_time = time.time() - self.start_time

        return len(
            self.initial_route), initial_dist, self.curr_dist, run_time, self.step, self.best_route, self.curr_route, self.best_dist


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


def worker(prob):
    try:
        file = "datasets/" + prob
        prob_name = prob.split('.')

        for cool in cooling_schedules:
            print("Handling problem %s.%s" % (prob, cool))

            # directory_path = '/Users/amberhawkins/Desktop/Thesis/Code/'
            directory_path = "/var/scratch/ahs354/SA_100_corrected/" + str(prob_name[0]) + "/"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            results_file = open(directory_path + cool + "_results.txt", "w")
            results_writer = csv.writer(results_file)
            results_writer.writerow(
                ['Problem', 'Number of Cities', 'Trial', 'Cooling Schedule', 'Initial Distance', 'Last Distance',
                 'Best Distance', 'Best Tour',
                 'Run Time'])

            # fitness_file = open(directory_path + cool + "_fitness.txt", "w")
            # fitness_writer = csv.writer(fitness_file)
            # fitness_writer.writerow(['Problem', 'Trial', 'Cooling Schedule', 'Iteration Number', 'Fitness'])

            for i in range(10):

                fitness_file = open(directory_path + str(i) + "_" + cool + "_fitness.txt", "w")
                fitness_writer = csv.writer(fitness_file)

                if i == 9:
                    fitness_writer.writerow(
                        ['Problem', 'Trial', 'Cooling Schedule', 'Iteration Number', 'Fitness', 'Tour'])

                    problem = SA(file, max_step=100000)
                    nr_cities, init_dist, curr_dist, run_time, nr_iter, best_route, curr_route, best_dist = problem.anneal(
                        cool)
                    print(
                        '{} - Prob: {}, Cooling: {}, Initial dist: {}, Best dist: {}, Last dist: {}, Run time:{}'.format(
                            i,
                            prob,
                            cool,
                            init_dist,
                            best_dist,
                            curr_dist,
                            run_time))
                    results_writer.writerow(
                        [prob_name[0], nr_cities, i, cool, init_dist, curr_dist, best_dist, best_route,
                         run_time])

                    fitness = problem.fitness_list

                    iteration = 1
                    for fit in fitness:
                        fitness_writer.writerow([prob_name[0], i, cool, iteration, fit, curr_route])
                        iteration += 1

                else:
                    fitness_writer.writerow(
                        ['Problem', 'Trial', 'Cooling Schedule', 'Iteration Number', 'Fitness'])

                    problem = SA(file, max_step=100000)
                    nr_cities, init_dist, curr_dist, run_time, nr_iter, best_route, curr_route, best_dist = problem.anneal(
                        cool)
                    print(
                        '{} - Prob: {}, Cooling: {}, Initial dist: {}, Best dist: {}, Last dist: {}, Run time:{}'.format(
                            i,
                            prob,
                            cool,
                            init_dist,
                            best_dist,
                            curr_dist,
                            run_time))

                    results_writer.writerow(
                        [prob_name[0], nr_cities, i, cool, init_dist, curr_dist, best_dist, best_route,
                         run_time])

                    fitness = problem.fitness_list

                    iteration = 1
                    for fit in fitness:
                        fitness_writer.writerow([prob_name[0], i, cool, iteration, fit])
                        iteration += 1


    except Exception as e:
        print("FAIL! Something went wrong with prob %s : %s" % (prob, e))
        print(traceback.format_exc())

    finally:
        # jobs.task_done() # Declare job is done
        print("Finished handling problem {}".format(prob))


if __name__ == '__main__':

    for prob in TSP_problems:
        p = Process(target=worker, args=(prob,))
        p.start()
