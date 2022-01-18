import json
import pandas as pd
import argparse
import time
import itertools
from chordToNote import ChordToNote
import pickle

with open("./modules/json_files/keychorddict.json") as f:
    data = json.load(f)
with open("./modules/pickle_files/key_chord_name_mapping.pickle", "rb") as f:
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
    input_idx,
    input_name,
    input_dict,
    chord_idx,
    chord_name,
    ed,
    length_match,
    chord,
    ismajor,
    hasSeventh,
):
    # print(chord, input_name, chord_name)
    score = 0
    for i, idx in enumerate(input_idx):
        if idx in chord_idx:
            score += 100
            score += 100 * input_dict[input_name[i]]
    for i, name in enumerate(input_name):
        if name in chord_name:
            score += 1000
            score += 1000 * input_dict[input_name[i]]
    # print(chord, score)
    if chord_name[0] in input_name:  # root is contained
        score += 500
        score += 100 * input_dict[chord_name[0]]
        if input_dict[chord_name[0]] == input_dict[min(input_dict, key=input_dict.get)]:
            score -= 50
    else:
        score -= 100
    # if chord_name[0] == input_name[0]:  # root is first
    #     score += 100 * input_dict[chord_name[0]]
    # score += 60 / (ed + 1)
    # if not length_match: #length match is not reliable when there are so many passing notes
    #     score -= 100
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
    if hasSeventh:
        score += 1
    return score


def MatchAnalysis(input_idx, input_name, chord_idx, chord_name, chord, key):
    idxMatch = intersection(input_idx, chord_idx)
    nameMatch = intersection(input_name, chord_name)
    if chord_name[0] in input_name:
        root_match = True
    else:
        root_match = False
    ed = edit_distance(input_idx, chord_idx)
    if len(input_idx) != len(chord_idx):
        length_match = False
    else:
        length_match = True
    seventhNote = ChordToNote(key, chord + "7")[-1]
    hasSeventh = seventhNote in input_name
    # print(key, chord, seventhNote, input_name, hasSeventh)
    return len(idxMatch), len(nameMatch), root_match, ed, length_match, hasSeventh


# ["a", "e", "b", "f#", "c#", "g#", "d", "g", "c", "f", "bb", "eb"]
# "Bb", "Eb", "Ab", "Db", "Gb"
key_mapping = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
changekey = {
    "GBMINOR": "F#MINOR",
    "DBMINOR": "C#MINOR",
    "ABMINOR": "G#MINOR",
    "A#MINOR": "BBMINOR",
    "D#MINOR": "EBMINOR",
    "A#MAJOR": "BBMAJOR",
    "D#MAJOR": "EBMAJOR",
    "G#MAJOR": "ABMAJOR",
    "C#MAJOR": "DBMAJOR",
    "F#MAJOR": "GBMAJOR",
}

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

    if keys[-1] == "-":
        return [key2num(key) for key in keys[:-1]]
    else:
        return [key2num(key) for key in keys]


major_root_offset = [0, 2, 4, 5, 7, 9, 11]  # Defining major scale as WWHWWWH
minor_root_offset = [0, 2, 3, 5, 7, 8, 10]  # Defining (natural) minor scale as WHWWHWW#
num_root_mapping = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}


def NoteToChord(keys_dict, key=None, numOut=10, threshold=0):
    """
        This is a weighted version.
        keys_dict will be a dictionary, notes as key, value as weight.
        Value should be normalized (add up to 1)
    """
    
    # removed, because it is possible to have length-1 segment from the RL module
    # if len(keys_dict) == 0:
    #     return None
    # elif len(keys_dict) == 1:
    #     onlynote = list(keys_dict.keys())[0]
    #     if onlynote[-2:] == "--":
    #         onlynote = onlynote[:-2] + "bb"
    #     elif onlynote[-1] == "-":
    #         onlynote = onlynote[:-1] + "b"
    #     if not key is None:
    #         note_idx = keys2num([onlynote])[0]
    #         key_idx = keys2num([key[:-5]])[0]
    #         if key.upper().find("MAJOR") != -1:
    #             ismajor = True
    #         else:
    #             ismajor = False
    #         offset = note_idx - key_idx
    #         offset = offset + 12 if offset < 0 else offset
    #         if ismajor:
    #             try:
    #                 idx = major_root_offset.index(offset) + 1
    #             except:
    #                 return None
    #         else:
    #             try:
    #                 idx = minor_root_offset.index(offset) + 1
    #             except:
    #                 return None
    #         return [{"Chord": key + num_root_mapping[idx]}]
    #     else:
    #         return [{"Chord": onlynote + "MajorI"}]

    if numOut is None:
        numOut = 10
    if threshold is None:
        threshold = 1
    if key is not None:
        key = key.upper()
    newkeydict = {}
    for dictkey in keys_dict:
        if dictkey[-2:] == "--":
            newdictkey = dictkey[:-2] + "bb"
            newkeydict[newdictkey] = keys_dict[dictkey]
        elif dictkey[-1] == "-":
            newdictkey = dictkey[:-1] + "b"
            newkeydict[newdictkey] = keys_dict[dictkey]
        else:
            newkeydict[dictkey] = keys_dict[dictkey]
    keys_dict = newkeydict
    keys_name = list(keys_dict.keys())
    keys_idx = keys2num(keys_name)
    sorted_keys = sorted(keys_idx)
    possible_chords = set()
    sorted_keys = sorted(list(set(keys_idx)))
    for i in range(threshold, 5):
        for each in itertools.combinations(sorted_keys, i):
            possible_chords.update(key_chord_name_mapping[str(each)])
    chords = list(possible_chords)
    if chords == []:
        return None
    # print(chords)
    rscore = []  # -1 for temp in range(len(chords))]
    rchord = []
    ridxMatch = []
    rnameMatch = []
    rrootMatch = []
    reditdist = []
    rlengthMatch = []
    hasSeventh = []
    numOk = 0
    if not key is None and key.upper() in changekey:
        key = changekey[key.upper()]
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
                isseventh,
            ) = MatchAnalysis(
                keys_idx,
                keys_name,
                entry["idx"],
                entry["naming"],
                entry["chord"],
                entry["key"],
            )
            score = ScoringModule(
                keys_idx,
                keys_name,
                keys_dict,  # for retrieving weight
                entry["idx"],
                entry["naming"],
                ed,
                length_match,
                entry["chord"],
                ismajor,
                isseventh,
            )
            rscore.append(score)
            rchord.append(chord)
            ridxMatch.append(idxMatch)
            rnameMatch.append(nameMatch)
            rrootMatch.append(rootMatch)
            reditdist.append(ed)
            rlengthMatch.append(length_match)
            hasSeventh.append(isseventh)
            numOk += 1
    if len(rscore) == 0:
        return None
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
    start = time.time()
    result = NoteToChord(
        {"B": 0.051},
        "Emajor",
    )
    end = time.time()
    print("Time taken:", end - start, "\n", result)

