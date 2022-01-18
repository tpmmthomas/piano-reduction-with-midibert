import argparse
import pickle
import json
import pandas as pd
import time

with open('pickle_files/key_chord_name_mapping.pickle','rb') as f:
  key_chord_name_mapping = pickle.load(f)

with open('json_files/keychorddict.json') as f:
    data = json.load(f)

key_mapping={
    'C':0,
    'D':2,
    'E':4,
    'F':5,
    'G':7,
    'A':9,
    'B':11
}

#1 by 1 key to number
def key2num(key):  
  key=key.upper()
  num=key_mapping[key[0]]
  modifier=len(key)
  if modifier==1:
    return num
  elif key[1]=='#':
    return (num+(modifier-1))%12
  elif key[1]=='B':
    return (num-(modifier-1))%12
  elif key[1]=='X':
    return (num+(modifier-1)*2)%12

# key_list to number_list
def keys2num(keys):
  if keys[-1]=='-':
    return [key2num(key) for key in keys[:-1]]
  else:
    return [key2num(key) for key in keys]



def intersection(a, b):
    temp = set(b)
    c = [value for value in a if value in temp]
    return c

# def edit_distance(a,b):
#     if len(a) > len(b):
#         a = a[:-1]
#     if len(b) > len(a):
#         b = b[:-1]
#     dist = 0
#     for i,val in enumerate(a):
#         dist += abs(val-b[i]) 
#     ###Scoring function
#     return 60//(dist+1)

def ScoringModule(input_idx,input_name,chord_idx,chord_name,chord):
    score = 0
    idxMatch = intersection(input_idx,chord_idx)
    score += 1000 * len(idxMatch)
    nameMatch = intersection(input_name,chord_name)
    score += 100 * len(nameMatch)
    #score += edit_distance(input_idx,chord_idx)
    if chord in ["I"]:
        score +=4
    elif chord in ["IV","V"]:
        score += 3
    elif chord in ["II","VI"]:
        score += 2
    elif chord in ["III","VII"]:
        score += 1
    if len(input_idx) != len(chord_idx):
        score -= 100
    return score

#print(key_chord_name_mapping)
def NoteToChordFast(keys_name,key=None,numOut=10):
  #result=[]
  keys_idx=keys2num(keys_name)
  sorted_keys = sorted(keys_idx)
  #print(keys)
  # for i in range(threshold,5):
  #   for each in itertools.combinations(keys,i):
  #     print(each)
  #     result.extend(key_chord_name_mapping[str(each)])
  chords = key_chord_name_mapping[str(tuple(sorted_keys))]
  chords2 = chords.copy()
  score = []
  for r in chords:
    entry = data[r]
    if key is not None and entry["key"].upper() != key.upper():
      chords2.remove(r)
      continue
    score.append(ScoringModule(keys_idx,keys_name,entry["idx"],entry["naming"],entry["chord"]))
  df = pd.DataFrame({"Chord":chords2,"Score":score})
  df = df.sort_values("Score",ascending=False)
  print("The most likely chords are:")
  print(df.head(numOut))
  return df.head(numOut)["Chord"].values

#TEST
# import time
# answer=''
# iterations=100000
# start = time.time()
# for i in range(iterations):
#   answer=keys2chords(['C','E','G'])
# end = time.time()
# print((end - start)/iterations)

# print(answer)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Output possible chords with given notes.")
    parser.add_argument("notes", nargs='+',help='The input keys (3 or 4 notes)')
    parser.add_argument("-o",'--numout',type=int,help='Number of output (optional)')
    parser.add_argument("-k","--key",help="The key (optional)")
    args = parser.parse_args()
    start = time.time()
    if args.numout is not None:
        NoteToChordFast(args.notes,args.key,args.numout)
    else:
        NoteToChordFast(args.notes,args.key)
    end = time.time()
    print("Time taken:",end-start)