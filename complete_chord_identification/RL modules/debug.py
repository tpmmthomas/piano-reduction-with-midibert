from env import SegmentationEnv
import random
import glob
training_pieces = []

for piece in glob.glob('./training/*'):
    training_pieces.append(piece)
# testing_pieces = []
# for piece in glob.glob('./testing/*'):
#     testing_pieces.append(piece)


env = SegmentationEnv(training_pieces)
for _ in range(19):
    print(random.randint(0,len(env.notes)-1))