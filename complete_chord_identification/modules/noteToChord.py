import json
import pandas as pd
import argparse
import time
import itertools
import pickle
from chordToNote import ChordToNote

with open("../modules/json_files/keychorddict.json") as f:
    data = json.load(f)
with open("../modules/pickle_files/key_chord_name_mapping.pickle", "rb") as f:
    key_chord_name_mapping = pickle.load(f)
for k in data:
    data[k]["key"] = data[k]["key"].upper()


def intersection(a, b):
    temp = set(b)
    c = [value for value in a if value in temp]
    return c


def edit_distance(a, b):
    if abs(len(a) - len(b)) >= 2:
        return 999
    if len(a) > len(b):
        a = a[:-1]
    if len(b) > len(a):
        b = b[:-1]
    dist = 0
    for i, val in enumerate(a):
        dist += abs(val - b[i])
    ###Scoring function
    return dist


def ScoringModule(
    idxMatch, nameMatch, rootMatch, ed, lengthMatch, chord, ismajor, root_first
):
    score = 0
    score += 1000 * nameMatch
    score += 100 * idxMatch
    if rootMatch:
        score += 500
    score += 60 / (ed + 1)
    if root_first:
        score += 10
    if ismajor:
        if chord in ["I"]:  # Tonic function chords
            score += 5
        elif chord == "VI":
            score += 4
        elif chord in ["IV", "II"]:  # Predominant function chords
            score += 3
        elif chord in ["V", "VII", "DimVII"]:  # Dominant function chords
            score += 2
    else:
        if chord in ["I", "VI"]:
            score += 4
        elif chord in ["IV", "II"]:  # Predominant function chords
            score += 3
        elif chord in ["V", "VII", "DimVII"]:  # Dominant function chords
            score += 2
    # if not lengthMatch:
    #     score -= 100
    if chord[-1] == "7":
        score -= 1
    return score


def MatchAnalysis(input_idx, input_name, chord_idx, chord_name, chord, key):
    idxMatch = intersection(input_idx, chord_idx)
    nameMatch = intersection(input_name, chord_name)
    if chord_name[0] in input_name:
        root_match = True
    else:
        root_match = False
    if chord_name[0] == input_name[0]:
        root_first = True
    else:
        root_first = False
    ed = edit_distance(input_idx, chord_idx)
    if len(input_idx) != len(chord_idx):
        length_match = False
    else:
        length_match = True
    seventhNote = ChordToNote(key, chord + "7")[-1]
    hasSeventh = seventhNote in input_name
    return (
        len(idxMatch),
        len(nameMatch),
        root_match,
        ed,
        length_match,
        root_first,
        hasSeventh,
    )


key_mapping = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# key_list to number_list
def keys2num(keys):
    # 1 by 1 key to number
    def key2num(key):
        key = key.upper()
        num = key_mapping[key[0]]
        modifier = len(key)
        if modifier == 1:
            return num
        elif key[1] == "#":
            return (num + (modifier - 1)) % 12
        elif key[1] == "B" or key[1] == "-":
            return (num - (modifier - 1)) % 12
        elif key[1] == "X":
            return (num + (modifier - 1) * 2) % 12

    #   if keys[-1] == "-":
    #      return [key2num(key) for key in keys[:-1]]
    #   else:
    return [key2num(key) for key in keys]


def NoteToChord(keys_name, key=None, numOut=10, threshold=2):

    if numOut is None:
        numOut = 10
    if threshold is None:
        threshold = 2
    if key is not None:
        key = key.upper()
    keys_name = [kn[:-1] + "b" if kn[-1] == "-" else kn for kn in keys_name]
    keys_idx = keys2num(keys_name)
    sorted_keys = sorted(list(set(keys_idx)))

    possible_chords = set()
    for i in range(threshold, 5):
        for each in itertools.combinations(sorted_keys, i):
            possible_chords.update(key_chord_name_mapping[str(each)])
    chords = list(possible_chords)
    if chords == []:
        return None, None

    rscore = []  # -1 for temp in range(len(chords))]
    rchord = []
    ridxMatch = []
    rnameMatch = []
    rrootMatch = []
    reditdist = []
    rlengthMatch = []
    hasSeventh = []
    numOk = 0
    for idx, chord in enumerate(chords):
        entry = data[chord]
        if (
            key is None or entry["key"] == key
        ):  ## remeber to make all key upper() after import**********\
            if entry["key"].upper().find("MAJOR") != -1:
                ismajor = True
            else:
                ismajor = False
            (
                idxMatch,
                nameMatch,
                rootMatch,
                ed,
                length_match,
                root_first,
                isSeventh,
            ) = MatchAnalysis(
                keys_idx,
                keys_name,
                entry["idx"],
                entry["naming"],
                entry["chord"],
                entry["key"],
            )
            score = ScoringModule(
                idxMatch,
                nameMatch,
                rootMatch,
                ed,
                length_match,
                entry["chord"],
                ismajor,
                root_first,
            )
            rscore.append(score)
            rchord.append(chord)
            ridxMatch.append(idxMatch)
            rnameMatch.append(nameMatch)
            rrootMatch.append(rootMatch)
            reditdist.append(ed)
            rlengthMatch.append(length_match)
            hasSeventh.append(isSeventh)
            numOk += 1

    (
        rscore,
        rchord,
        ridxMatch,
        rnameMatch,
        rrootMatch,
        reditdist,
        rlengthMatch,
        hasSeventh,
    ) = zip(
        *sorted(
            zip(
                rscore,
                rchord,
                ridxMatch,
                rnameMatch,
                rrootMatch,
                reditdist,
                rlengthMatch,
                hasSeventh,
            ),
            reverse=True,
        )[: min(numOk, numOut)]
    )

    # print(rchord,rscore)

    # df = pd.DataFrame({"Chord":rchord,"Score":rscore,"pitch match":ridxMatch,"name match":rnameMatch,"root present":rrootMatch,"edit distance":reditdist,"length match":rlengthMatch})
    # df = df.sort_values("Score",ascending=False)
    # score,chords=zip(*sorted(zip(score,chords),reverse=True)[:min(numOk,numOut)])

    # format output
    result = []
    for idx in range(len(rscore)):
        result.append(
            {
                "Chord": rchord[idx],
                "Score": rscore[idx],
                "pitch match": ridxMatch[idx],
                "name match": rnameMatch[idx],
                "root present": rrootMatch[idx],
                "edit distance": reditdist[idx],
                "length match": rlengthMatch[idx],
                "hasSeventh": hasSeventh[idx],
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Output possible chords with given notes."
    )
    parser.add_argument("notes", nargs="+", help="The input keys (3 or 4 notes)")
    parser.add_argument("-o", "--numout", type=int, help="Number of output (optional)")
    parser.add_argument("-k", "--key", help="The key (optional)")
    parser.add_argument(
        "-t", "--threshold", type=int, help="Least number of key matches (optional)"
    )
    args = parser.parse_args()
    start = time.time()
    result = NoteToChord(args.notes, args.key, args.numout, args.threshold)
    end = time.time()
    print("Time taken:", end - start, "\n", result)

