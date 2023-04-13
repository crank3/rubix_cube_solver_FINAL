
# This version has two separate networks for the value and policy.
# The commented out print statements likely have not been updated for a bit.
# The target value in ADI is based on the distance from solved cube.



import random
import math
import time
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



class Cube2x2:

    def __init__(self, state=None):
        '''
        ================ STATE ================



        ['B','B','B','B','O','O','O','O','R','R','R','R','Y','Y','Y','Y','W','W','W','W','G','G','G','G']

                   YY
                   YY
                OO BB RR GG
                OO BB RR GG
                   WW
                   WW


        [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x]

                   mn
                   op
                ef ab ij uv
                gh cd kl wx
                   qr
                   st



        In order of least to greatest numbers:
        Faces:
            [4]
        [2] [1] [3] [6]
            [5]
        Tiles:
                [1][2]
                [3][4]

        [1][2]  [1][2]  [1][2]  [1][2]
        [3][4]  [3][4]  [3][4]  [3][4]

                [1][2]
                [3][4]

        '''

        self.state = state


        if self.state is None:
            self.state = ['B','B','B','B','O','O','O','O','R','R','R','R','Y','Y','Y','Y','W','W','W','W','G','G','G','G']


    def isSolved(self):

        face1 = self.state[:4]
        face2 = self.state[4:8]
        face3 = self.state[8:12]
        face4 = self.state[12:16]
        face5 = self.state[16:20]
        face6 = self.state[20:]

        for color in face1:
            if color != face1[0]:
                return False
        for color in face2:
            if color != face2[0]:
                return False
        for color in face3:
            if color != face3[0]:
                return False
        for color in face4:
            if color != face4[0]:
                return False
        for color in face5:
            if color != face5[0]:
                return False
        for color in face6:
            if color != face6[0]:
                return False
        return True

    def move(self, turn):
        # only using F, Fp, U, Up, R, Rp

        temp_state = self.state.copy()

        if turn == 'F':
            self.state[0] = temp_state[2]
            self.state[1] = temp_state[0]
            self.state[3] = temp_state[1]
            self.state[2] = temp_state[3]
            self.state[14] = temp_state[7]
            self.state[15] = temp_state[5]
            self.state[8] = temp_state[14]
            self.state[10] = temp_state[15]
            self.state[17] = temp_state[8]
            self.state[16] = temp_state[10]
            self.state[7] = temp_state[17]
            self.state[5] = temp_state[16]

        elif turn == 'Fp':
            self.state[0] = temp_state[1]
            self.state[1] = temp_state[3]
            self.state[3] = temp_state[2]
            self.state[2] = temp_state[0]
            self.state[14] = temp_state[8]
            self.state[15] = temp_state[10]
            self.state[8] = temp_state[17]
            self.state[10] = temp_state[16]
            self.state[17] = temp_state[7]
            self.state[16] = temp_state[5]
            self.state[7] = temp_state[14]
            self.state[5] = temp_state[15]

        elif turn == 'L':
            self.state[4] = temp_state[6]
            self.state[5] = temp_state[4]
            self.state[7] = temp_state[5]
            self.state[6] = temp_state[7]
            self.state[12] = temp_state[23]
            self.state[14] = temp_state[21]
            self.state[0] = temp_state[12]
            self.state[2] = temp_state[14]
            self.state[16] = temp_state[0]
            self.state[18] = temp_state[2]
            self.state[23] = temp_state[16]
            self.state[21] = temp_state[18]

        elif turn == 'Lp':
            self.state[4] = temp_state[5]
            self.state[5] = temp_state[7]
            self.state[7] = temp_state[6]
            self.state[6] = temp_state[4]
            self.state[12] = temp_state[0]
            self.state[14] = temp_state[2]
            self.state[0] = temp_state[16]
            self.state[2] = temp_state[18]
            self.state[16] = temp_state[23]
            self.state[18] = temp_state[21]
            self.state[23] = temp_state[12]
            self.state[21] = temp_state[14]

        elif turn == 'R':
            self.state[8] = temp_state[10]
            self.state[9] = temp_state[8]
            self.state[11] = temp_state[9]
            self.state[10] = temp_state[11]
            self.state[15] = temp_state[3]
            self.state[13] = temp_state[1]
            self.state[20] = temp_state[15]
            self.state[22] = temp_state[13]
            self.state[19] = temp_state[20]
            self.state[17] = temp_state[22]
            self.state[3] = temp_state[19]
            self.state[1] = temp_state[17]

        elif turn == 'Rp':
            self.state[8] = temp_state[9]
            self.state[9] = temp_state[11]
            self.state[11] = temp_state[10]
            self.state[10] = temp_state[8]
            self.state[15] = temp_state[20]
            self.state[13] = temp_state[22]
            self.state[20] = temp_state[19]
            self.state[22] = temp_state[17]
            self.state[19] = temp_state[3]
            self.state[17] = temp_state[1]
            self.state[3] = temp_state[15]
            self.state[1] = temp_state[13]

        elif turn == 'U':
            self.state[12] = temp_state[14]
            self.state[13] = temp_state[12]
            self.state[15] = temp_state[13]
            self.state[14] = temp_state[15]
            self.state[21] = temp_state[5]
            self.state[20] = temp_state[4]
            self.state[9] = temp_state[21]
            self.state[8] = temp_state[20]
            self.state[1] = temp_state[9]
            self.state[0] = temp_state[8]
            self.state[5] = temp_state[1]
            self.state[4] = temp_state[0]

        elif turn == 'Up':
            self.state[12] = temp_state[13]
            self.state[13] = temp_state[15]
            self.state[15] = temp_state[14]
            self.state[14] = temp_state[12]
            self.state[21] = temp_state[9]
            self.state[20] = temp_state[8]
            self.state[9] = temp_state[1]
            self.state[8] = temp_state[0]
            self.state[1] = temp_state[5]
            self.state[0] = temp_state[4]
            self.state[5] = temp_state[21]
            self.state[4] = temp_state[20]

        elif turn == 'D':
            self.state[16] = temp_state[18]
            self.state[17] = temp_state[16]
            self.state[19] = temp_state[17]
            self.state[18] = temp_state[19]
            self.state[2] = temp_state[6]
            self.state[3] = temp_state[7]
            self.state[10] = temp_state[2]
            self.state[11] = temp_state[3]
            self.state[22] = temp_state[10]
            self.state[23] = temp_state[11]
            self.state[6] = temp_state[22]
            self.state[7] = temp_state[23]

        elif turn == 'Dp':
            self.state[16] = temp_state[17]
            self.state[17] = temp_state[19]
            self.state[19] = temp_state[18]
            self.state[18] = temp_state[16]
            self.state[2] = temp_state[10]
            self.state[3] = temp_state[11]
            self.state[10] = temp_state[22]
            self.state[11] = temp_state[23]
            self.state[22] = temp_state[6]
            self.state[23] = temp_state[7]
            self.state[6] = temp_state[2]
            self.state[7] = temp_state[3]

        elif turn == 'B':
            self.state[20] = temp_state[22]
            self.state[21] = temp_state[20]
            self.state[23] = temp_state[21]
            self.state[22] = temp_state[23]
            self.state[13] = temp_state[11]
            self.state[12] = temp_state[9]
            self.state[4] = temp_state[13]
            self.state[6] = temp_state[12]
            self.state[18] = temp_state[4]
            self.state[19] = temp_state[6]
            self.state[11] = temp_state[18]
            self.state[9] = temp_state[19]

        elif turn == 'Bp':
            self.state[20] = temp_state[21]
            self.state[21] = temp_state[23]
            self.state[23] = temp_state[22]
            self.state[22] = temp_state[20]
            self.state[13] = temp_state[4]
            self.state[12] = temp_state[6]
            self.state[4] = temp_state[18]
            self.state[6] = temp_state[19]
            self.state[18] = temp_state[11]
            self.state[19] = temp_state[9]
            self.state[11] = temp_state[13]
            self.state[9] = temp_state[12]
    def testSequence(self, sequence):

        temp_cube = Cube2x2(self.state.copy())  # .copy() is important!!!

        for turn in sequence:
            temp_cube.move(turn)

        return temp_cube.isSolved()



    def bruteSolve(self):

        if self.isSolved():
            return 'Already solved'

        moves = ['F', 'Fp', 'U', 'Up', 'R', 'Rp']

        for i1 in moves:
            if self.testSequence([i1]): return [i1]
        print('Move length 1 completed')

        for i1 in moves:
            for i2 in moves:
                if self.testSequence([i1,i2]): return [i1,i2]
        print('Move length 2 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    if self.testSequence([i1,i2,i3]): return [i1,i2,i3]
        print('Move length 3 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        if self.testSequence([i1,i2,i3,i4]): return [i1,i2,i3,i4]
        print('Move length 4 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            if self.testSequence([i1,i2,i3,i4,i5]): return [i1,i2,i3,i4,i5]
        print('Move length 5 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                if self.testSequence([i1,i2,i3,i4,i5,i6]): return [i1,i2,i3,i4,i5,i6]
        print('Move length 6 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    if self.testSequence([i1,i2,i3,i4,i5,i6,i7]): return [i1,i2,i3,i4,i5,i6,i7]
        print('Move length 7 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8]): return [i1,i2,i3,i4,i5,i6,i7,i8]
        print('Move length 8 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9]
        print('Move length 9 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10]
        print('Move length 10 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]
        print('Move length 11 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12]
        print('Move length 12 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        for i13 in moves:
                                                            if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13]
        print('Move length 13 completed')

        for i1 in moves:
            for i2 in moves:
                for i3 in moves:
                    for i4 in moves:
                        for i5 in moves:
                            for i6 in moves:
                                for i7 in moves:
                                    for i8 in moves:
                                        for i9 in moves:
                                            for i10 in moves:
                                                for i11 in moves:
                                                    for i12 in moves:
                                                        for i13 in moves:
                                                            for i14 in moves:
                                                                if self.testSequence([i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14]): return [i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14]
        return 'Error: no solution found'

    def randomSolve(self):

        if self.isSolved():
            return 'Already solved'

        moves = ['F', 'Fp', 'U', 'Up', 'R', 'Rp']

        while True:

            rand_length = random.randint(1, 14)
            sequence = []

            for i in range(rand_length):
                sequence.append(moves[random.randint(0, 5)])

            if self.testSequence(sequence):
                return sequence

    def MCTS_solve(self, policy_model, value_model, minutes):

        if self.isSolved():
            return 'Already solved'

        c = 1
        v = 0

        print('node: you probably want to set v to something other than zero eventually.')

        tree = [ [Node(self.state, P=policy_model.predict(np.array([encodeOneHot(self.state)]))[0])] ]
        current_pos = [0,0]
        current_node = tree[0][0]
        # The structure of the tree can be thought of as [generation][move], except the moves are added for when the
        # children of multiple cubes are in a single generation.

        actions = ['F', 'Fp', 'U', 'Up', 'R', 'Rp']
        seconds = minutes * 60
        start = time.time()

        solve_time = 0

        total_max_value = 0
        best_cube = None

        while (time.time() - start) < seconds:
            print(f'------------------------------------------------------------------------------------------')

            print(f'current_pos: {current_pos}')
            print(f'tree length: {len(tree)}')

            if current_node.children == None:
                print('==================== Adding Children ====================')

                new_children = []

                for action in actions:

                    child_cube = Cube2x2(current_node.state.copy())
                    child_cube.move(action)
                    # print(f'child_cube: {child_cube.state}')

                    children_index = 0

                    if current_pos[0] < len(tree) - 1:
                        children_index = len(tree[current_pos[0] + 1])

                    current_node.children = range(children_index, children_index + 6)

                    if child_cube.isSolved():
                        print('SOLVED CUBE ADDED *********')
                        if solve_time == 0:
                            solve_time = time.time() - start
                            print(f'solve_time = {solve_time}')

                    new_children.append(Node(child_cube.state.copy(), P=policy_model.predict(np.array([encodeOneHot(child_cube.state)]))[0], parent=current_pos[1], previous_action=action))

                    print(f'Child {action} Policy:  {new_children[len(new_children) - 1].P}')

                if current_pos[0] < len(tree) - 1:
                    tree[current_pos[0] + 1] += new_children
                else:
                    tree.append(new_children)

                max_value = -1000
                for i in current_node.children:
                    value = value_model.predict(np.array([encodeOneHot(tree[current_pos[0] + 1][i].state)]))[0][0]

                    if value > max_value:
                        max_value = value
                        print(f'max_value ----> {max_value}')

                    if max_value > total_max_value:
                        total_max_value = max_value
                        print(f'total_max_value ----> {total_max_value}')

                        best_cube = Cube2x2(tree[current_pos[0] + 1][i].state.copy())

                while current_node.parent != None:  # this loop iterates through every cube visited in the current path
                    action_i = actions.index(current_node.previous_action)

                    current_pos = [current_pos[0] - 1, current_node.parent]
                    current_node = tree[current_pos[0]][current_pos[1]]

                    if current_node.W[action_i] < max_value:
                        current_node.W[action_i] = max_value

                    current_node.N[action_i] += 1
                    current_node.L[action_i] -= v

            else:
                print('==================== Selecting Action ====================')

                action_i = 0
                max = -10000

                for i in range(len(actions)):

                    # U(a)
                    summation = 0

                    for j in range(len(actions)):
                        summation += current_node.N[j]
                        # Looking back at the article makes me think this is supposed to be the sum of N(a) for all of
                        # the moves leading up to the current one. Actually i think im wrong.

                    # U = c * current_node.P[i] * (math.sqrt(summation) / (1 + current_node.N[i]) )  # dont work

                    # U = c * (   current_node.P[i] + ((math.sqrt(summation) / (1 + current_node.N[i])))   )

                    # U = c * (current_node.P[i] + ((1 / (1 + current_node.N[i]))))

                    # U = c * current_node.P[i] - (current_node.N[i]/20)

                    U = c * current_node.P[i] * ((1 + math.sqrt(summation)) / (1 + current_node.N[i]))
                    Q = current_node.W[i] - current_node.L[i]

                    print(f'--------- {actions[i]} ---------')
                    print(f'Policy:  {round(1*(current_node.P[i]), 5)}')  # print this move's probability
                    print(f'U = {round(U, 3)}, Q = {round(Q, 3)},  U+Q = {round(U+Q, 3)}')

                    if U + Q > max:
                        max = U + Q
                        action_i = i



                print(f'============ Selected Move: {actions[action_i]}')

                current_node.L[action_i] += v
                current_pos = [current_pos[0] + 1, current_node.children[action_i]]
                current_node = tree[current_pos[0]][current_pos[1]]

                if Cube2x2(current_node.state).isSolved():

                    while current_node.parent != None:
                        temp_action_i = actions.index(current_node.previous_action)
                        current_pos = [current_pos[0] - 1, current_node.parent]
                        current_node = tree[current_pos[0]][current_pos[1]]
                        current_node.N[temp_action_i] += 1
            print(f'--------------------------------------------', end='\n\n\n')

        for row in range(1, len(tree)):
            for node in range(len(tree[row])):
                if Cube2x2(tree[row][node].state).isSolved():
                    current_pos = [row, node]
                    current_node = tree[row][node]

                    solution = []
                    while current_node.previous_action != None:
                        solution.insert(0, current_node.previous_action)
                        current_pos = [current_pos[0] - 1, current_node.parent]
                        current_node = tree[current_pos[0]][current_pos[1]]

                    print(f'solve_time = {solve_time}')
                    return solution

        print(f'best_cube: ', end='')
        print(*best_cube.state, sep='', end='\n')
        print(f'total_max_value: {total_max_value}')

        return 'No solution found'


    def greedy_solve(self, policy_model, minutes):
        '''
        this is basically the "greedy" solver talked about in the 2x2 article: <https://web.stanford.edu/class/aa228/reports/2018/final28.pdf>
        '''

        current_cube = Cube2x2(self.state.copy())

        actions = ['F', 'Fp', 'U', 'Up', 'R', 'Rp']
        seconds = minutes * 60
        start = time.time()

        previous_moves = []
        while (time.time() - start) < seconds:
            if current_cube.isSolved():
                print(f'SOLVED CUBE!!!!!!!')
            policy_out = policy_model.predict(np.array([encodeOneHot(current_cube.state)]))

            # policy_output_list.remove(inverseMove(previous_move))  # pain

            if not previous_moves:  # if "previous_moves" is an empty list
                best_move = maxPolicyMove(policy_out)
            else:  # if it is not empty
                best_move = maxPolicyMove(policy_out, [inverseMove(previous_moves[-1])])  # cant chose inverse of last move

                if len(previous_moves) >= 4:
                    if (previous_moves[-1] == previous_moves[-2]) and (previous_moves[-1] == previous_moves[-3]) and (previous_moves[-1] == previous_moves[-4]):
                        print(f'LOOPED')
                        best_move = maxPolicyMove(policy_out, [inverseMove(previous_moves[-1]), previous_moves[-1]])

            print(f'chosen move: {best_move}')
            current_cube.move(best_move)

            previous_moves.append(best_move)

class Node:

    def __init__(self, state=None, N=None, W=None, L=None, P=None, parent=None, children=None, previous_action=None):
        self.state = state
        self.N = N
        self.W = W
        self.L = L
        self.P = P
        self.parent = parent
        self.children = children
        self.previous_action = previous_action

        if self.state is None:
            self.state = ['B','B','B','B','O','O','O','O','R','R','R','R','Y','Y','Y','Y','W','W','W','W','G','G','G','G']
        if self.N is None:
            self.N = [0,0,0,0,0,0]
        if self.W is None:
            self.W = [0,0,0,0,0,0]
        if self.L is None:
            self.L = [0,0,0,0,0,0]
        if self.P is None:
            self.P = [0,0,0,0,0,0]


def maxPolicyMove(policy_output, excluding=None):
    '''
    :param policy_output: The output of the policy network from model.predict(). It is a numpy list of six elements.
    :param excluding: A list of strings that represent the moves which should not be returned by this function.
    :return: Returns the string representing the best move according to the policy network's output.
    '''
    moves = ['F', 'Fp', 'U', 'Up', 'R', 'Rp']
    policy_output_list = []
    for element in policy_output[0]:  # loop that makes a regular python list of "policy_output"
        policy_output_list.append(element)

    if excluding is not None:
        for excluded_element in excluding:  # loop that sets the policies of the excluded moves to 0
            if excluded_element is not None:
                policy_output_list[moves.index(excluded_element)] = 0

    max_output = max(policy_output_list)
    index_max = policy_output_list.index(max_output)
    return moves[index_max]


def randomMove(excluding=None):
    excluded = excluding
    # returns a random element from the list "moves" excluding the elements of the list argument "excluding"
    moves = ['F', 'Fp', 'U', 'Up', 'R', 'Rp']

    if excluded is None:
        return moves[random.randint(0, len(moves)-1)]
    else:
        for move in excluded:
            moves.remove(move)

        return moves[random.randint(0, len(moves)-1)]


def inverseMove(move):
    # returns the move that undoes the argument "move"
    inverse = ''
    if move == 'F':
        inverse = 'Fp'
    elif move == 'Fp':
        inverse = 'F'
    elif move == 'R':
        inverse = 'Rp'
    elif move == 'Rp':
        inverse = 'R'
    elif move == 'U':
        inverse = 'Up'
    elif move == 'Up':
        inverse = 'U'
    else:
        return None
    return inverse


def randomCube(moves_away):
    random_cube = Cube2x2()
    sequence_length = moves_away
    sequence = []
    random_move = ''
    for i in range(sequence_length):  # tries to create a sequence where every move is further from the cube than the last
        if sequence:
            if len(sequence) >= 2:
                if sequence[-1] == sequence[-2]:
                    random_move = randomMove([inverseMove(sequence[-1]), sequence[-1]])
                else:
                    random_move = randomMove([inverseMove(sequence[-1])])
        else:
            random_move = randomMove()
        sequence.append(random_move)
        random_cube.move(random_move)  # used to be:   cube.move(moves[random.randint(0, 11)])
    print(f'sequence = {sequence}')
    return random_cube


def encodeOneHot(state):

    one_hot = []

    for s in state:
        if s == 'B':
            one_hot += [1,0,0,0,0,0]
        elif s == 'O':
            one_hot += [0,1,0,0,0,0]
        elif s == 'R':
            one_hot += [0,0,1,0,0,0]
        elif s == 'Y':
            one_hot += [0,0,0,1,0,0]
        elif s == 'W':
            one_hot += [0,0,0,0,1,0]
        elif s == 'G':
            one_hot += [0,0,0,0,0,1]

    return one_hot

def decodeOneHot(one_hot):

    state = ''

    start = 0
    end = 6

    while end <= len(one_hot):

        section = one_hot[start:end]

        if section == '100000':
            state += 'B'
        elif section == '010000':
            state += 'O'
        elif section == '001000':
            state += 'R'
        elif section == '000100':
            state += 'Y'
        elif section == '000010':
            state += 'W'
        elif section == '000001':
            state += 'G'

        start += 6
        end += 6

    return state

def ADI(minutes, models=None):
    # the "models" argument is a list that looks like [policy_network, value_network]

    if models == None:

        # Creating the policy network
        policy_model = Sequential()
        policy_model.add(Dense(1000, input_dim=144, activation='relu'))
        policy_model.add(Dense(1000, activation='relu'))
        policy_model.add(Dense(6, activation='softmax'))

        policy_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


        # Creating the value network
        value_model = Sequential()
        value_model.add(Dense(500, input_dim=144, activation='relu'))
        value_model.add(Dense(500, activation='relu'))
        value_model.add(Dense(1, activation='relu'))

        value_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    else:
        policy_model = models[0]
        value_model = models[1]

    # Training Process

    moves = ['F', 'Fp', 'U', 'Up', 'R', 'Rp']
    seconds = minutes * 60
    start = time.time()

    total_samples = 0

    loss_print = []

    while (time.time() - start) < seconds:

        cube = Cube2x2()
        training_inputs = []

        sequence_length = 14

        sequence = []

        for i in range(sequence_length):  # tries to create a sequence where every move is further from the cube than the last
            if sequence:
                if len(sequence) >= 2:
                    if sequence[-1] == sequence[-2]:
                        random_move = randomMove([inverseMove(sequence[-1]), sequence[-1]])
                    else:
                        random_move = randomMove([inverseMove(sequence[-1])])
            else:
                random_move = randomMove()
            sequence.append(random_move)
            cube.move(random_move)  # used to be:   cube.move(moves[random.randint(0, 11)])
            training_inputs.append(Cube2x2(cube.state.copy()))

        print(f'\n========================================================================================================================================================== NEW SEQUENCE')
        print(f'sequence: ', end='')
        for move in sequence:
            print(f'{move}', end='')
        print(f'\n')

        total_samples += sequence_length

        for i in range(len(training_inputs)):
            print(f'\n------------------------------------------------------ CUBE {i}  ', end='')
            print(*training_inputs[i].state, sep='')

            value_target = -100
            policy_target_move = None

            parent_of_solved = False  # represents whether the state (i) is a parent to the solved state

            for move in moves:
                child = Cube2x2(training_inputs[i].state.copy())
                child.move(move)
                child_value = sequence_length - i  # used to be value_estimate
                # child_value is sequence_length - (distance of the child to the solved cube), and it will act as the value target for the network.
                # child_value should be highest when the child is the solved cube, and its highest value is equal to the sequence length

                print(f'...................... CHILD {move}:  ', end='')
                print(*child.state, sep='')

                if move == inverseMove(sequence[i]):
                    print(f'INVERSE (child_value += 1)')
                    child_value += 1

                if child.isSolved():
                    print(f'SOLVED (child_value = 15)')
                    parent_of_solved = True
                    child_value = 15

                print(f'child_value = {child_value}')

                if child_value > value_target:
                    value_target = child_value
                    policy_target_move = move
                    print(f'value_target -----> {value_target}')
                    print(f'policy_target -----> {policy_target_move}')

            value_target /= 15  # makes the min value target 0 and the max 1

            policy_target = None
            if policy_target_move == 'F':
                policy_target = [1,0,0,0,0,0]   # CHANGED THIS (removed the extra elements in the array that represented the moves that i removed)
            elif policy_target_move == 'Fp':
                policy_target = [0,1,0,0,0,0]
            elif policy_target_move == 'U':
                policy_target = [0,0,1,0,0,0]
            elif policy_target_move == 'Up':
                policy_target = [0,0,0,1,0,0]
            elif policy_target_move == 'R':
                policy_target = [0,0,0,0,1,0]
            elif policy_target_move == 'Rp':
                policy_target = [0,0,0,0,0,1]

            # determines the training weight of the sample

            # sample_weight = 1  # constant sample weight.
            # sample_weight = (1 / (i + 1))  # assigns more weight the closer the parent is to the solved cube (THIS ONE BREAKS THE VALUE NETWORK)
            # sample_weight = (-(i/16))+1  # Linear. y=0 when x=16.
            sample_weight = (1/4)*(((-i)+16)**(1/2))  # upside-down square root. y=0 when x=16.
            # sample_weight = (-1/256)*(i**2)+1  # upside-down parabola. y=0 when x=16.

            '''
            if parent_of_solved:
                sample_weight += 2  # assigns more weight to parent of solved cube
            '''


            #####################################################
            # Printing Training (prints loss, target policy of each sample, and the output of the network as it trains)

            print(f'-----------\nsample_weight: {sample_weight}')

            # policy network


            target = []
            for element in policy_target:
                target.append(element)
            print(f'policy target: {target}')

            output_temp = policy_model.predict(np.array([encodeOneHot(training_inputs[i].state)]))
            actual = []
            for element in output_temp[0]:
                actual.append(element)

            print(f'policy actual: [', end='')
            for i_index in range(len(actual)):
                print(f'{str(round(actual[i_index], 3))}', end='')
                if actual[i_index] == max(actual):
                    print(f'■', end='')
                if i_index != len(actual)-1:
                    print(f', ', end='')
            print(f']', end='\n')
            
            
            
            # value network
            target = value_target
            print(f'value target: {target}')

            output_temp = value_model.predict(np.array([encodeOneHot(training_inputs[i].state)]))
            print(f'value actual: {output_temp[0]}')

            loss = tf.keras.losses.mean_squared_error(target, output_temp[0])
            loss_print.append(loss.numpy())
            # loss_print.append(loss.numpy() * sample_weight)



            #####################################################

            policy_model.fit(np.array([encodeOneHot(training_inputs[i].state)]), np.array([policy_target]), sample_weight=np.array([sample_weight]))

            value_model.fit(np.array([encodeOneHot(training_inputs[i].state)]), np.array([value_target]), sample_weight=np.array([sample_weight]))

            print(f'-----------')

    '''
    for loss in loss_print:
        loss_rounded = str(round(loss, 5))
        if loss_rounded.find("e") != -1:
            print('0')
        else:
            print(loss_rounded)
    print('Total Samples: ', total_samples)
    '''

    return [policy_model, value_model]


#################################
# just printing stuff

print(f'Solved Cube = BBBBOOOORRRRYYYYWWWWGGGG', end='\n\n')


cube_test = Cube2x2()
cube_test.move('Up')
print(f'cube after Up:  ', end='')
print(*cube_test.state, sep='', end='\n\n')

cube_test.move('R')
print(f'cube after Up R:  ', end='')
print(*cube_test.state, sep='', end='\n\n')

cube_test.move('Up')
print(f'cube after Up R Up:  ', end='')
print(*cube_test.state, sep='', end='\n\n')

cube_test.move('Rp')
print(f'cube after Up R Up Rp:  ', end='')
print(*cube_test.state, sep='', end='\n\n')

cube_test.move('F')
print(f'cube after Up R Up Rp F:  ', end='')
print(*cube_test.state, sep='', end='\n\n')



random_cube = randomCube(7)
print(f'random_cube (7 moves away): ', end='')
print(*random_cube.state, sep='', end='\n\n')

#################################




input_state = input('\nInput the cube state: ')

start = time.time()

input_cube = Cube2x2(list(input_state))
print(input_cube.state)

print('\n--------AUTODIDACTIC ITERATION--------\n')
# models = ADI(115)
# models = ADI(5, [tf.keras.models.load_model('policy_model'), tf.keras.models.load_model('value_model')])
models = [tf.keras.models.load_model('policy_model'), tf.keras.models.load_model('value_model')]

# models[0].save('policy_model')
# models[1].save('value_model')


print('\n--------GREEDY--------\n')
# input_cube.greedy_solve(models[0], 1)

print('\n--------MONTE CARLO TREE SEARCH--------\n')
solution = input_cube.MCTS_solve(models[0], models[1], 0.5)

print(*input_cube.state, sep='', end='\n')
print('--------------------------\nSolution: ', solution, '\n--------------------------')

end = time.time()
print((end - start), 'seconds')
