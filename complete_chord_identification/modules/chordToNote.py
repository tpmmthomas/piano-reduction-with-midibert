import argparse

# parameters
major_offset = [0, 4, 7, 10]  # 0 is the root of the chord
minor_offset = [0, 3, 7, 10]
major_seventh_offset = [
    1,
    0,
    0,
    1,
    0,
    0,
    1,
]  # Determines whether an additional half-step is needed for chords I to VII
minor_seventh_offset = [0, 1, 1, 0, 0, 0, 0]
diminished_offset = [0, 3, 6, 9]
augmented_offset = [0, 4, 8, 10]
german_offset = [0, 4, 7, 10]
french_offset = [0, 4, 6, 10]
italian_offset = [0, 4, 10]
# Using chart in shared file as reference
major_key_Chordtype = [0, 1, 1, 0, 0, 1, 2]
minor_key_Chordtype = [1, 2, 0, 1, 1, 0, 0]
key_map = {0: "maj", 1: "min", 2: "dim"}
major_root_offset = [0, 2, 4, 5, 7, 9, 11]  # Defining major scale as WWHWWWH
minor_root_offset = [0, 2, 3, 5, 7, 8, 10]  # Defining (natural) minor scale as WHWWHWW
# Following are different enharmonic mappings from index to pitch.
pitch_to_index = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
index_to_pitch_sharp = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}
index_to_pitch_flat = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "Gb",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}
index_to_pitch_doublesharp = {
    0: "B#",
    1: "C#",
    2: "Cx",
    3: "D#",
    4: "Dx",
    5: "E#",
    6: "F#",
    7: "Fx",
    8: "G#",
    9: "Gx",
    10: "A#",
    11: "Ax",
}
index_to_pitch_doubleflat = {
    0: "Dbb",
    1: "Db",
    2: "Ebb",
    3: "Eb",
    4: "Fb",
    5: "Gbb",
    6: "Gb",
    7: "Abb",
    8: "Ab",
    9: "Bbb",
    10: "Bb",
    11: "Cb",
}
relative_major = {
    "A": "C",
    "E": "G",
    "B": "D",
    "F#": "A",
    "C#": "E",
    "G#": "B",
    "EB": "GB",
    "BB": "DB",
    "F": "AB",
    "C": "EB",
    "G": "BB",
    "D": "A",
}

# Input chord format: I to VII
# Possible suffix:  +, - and 7, +/- precedes 7
# Possible prefix: german as "ger", french as "fre", italian as "ita",
# diminish as "dim", augmented as "aug" ,flat as "b", +/- suffix not available if contained these prefix

"""
def InputValidation(key,chord):
    if key[-5:].upper() != "MAJOR" and key[-5:].upper() != "MINOR":
        print("Invalid key.")
        return False
    if len(chord) >=6 and chord[:6].upper() == "GERMAN":
        chord = chord[-6:]
    if len(chord) >=6 and chord[:6].upper() == "FRENCH":
        chord = chord[-6:]
    if len(chord) >=7 and chord[:7].upper() == "ITALIAN":
        chord = chord[-7:]
"""


def RomanToInt(x):  # only check from beginning, ignore suffix
    if len(x) >= 3 and x[:3].upper() == "III":
        return 3
    elif len(x) >= 3 and x[:3].upper() == "VII":
        return 7
    elif len(x) >= 2 and x[:2].upper() == "II":
        return 2
    elif len(x) >= 2 and x[:2].upper() == "IV":
        return 4
    elif len(x) >= 2 and x[:2].upper() == "VI":
        return 6
    elif x[0].upper() == "I":
        return 1
    elif x[0].upper() == "V":
        return 5
    else:
        return 0


# Determines the chord type (major,minor,etc)
def typeAnalysis(key, chord):
    isSeven = chord[-1:] == "7"
    if isSeven:
        chord = chord[:-1]
    if len(chord) >= 3 and chord[0:3].upper() == "GER":
        return "ger", isSeven
    elif len(chord) >= 3 and chord[0:3].upper() == "FRE":
        return "fre", isSeven
    elif len(chord) >= 3 and chord[0:3].upper() == "ITA":
        return "ita", isSeven
    elif len(chord) >= 3 and chord[0:3].upper() == "DIM":
        return "dim", isSeven
    elif len(chord) >= 3 and chord[0:3].upper() == "AUG":
        return "aug", isSeven
    elif chord[0].upper() == "B":
        return "maj", isSeven
    elif chord[-1:] == "+":
        return "maj", isSeven
    elif chord[-1:] == "-":
        return "min", isSeven
    else:
        if key[-5:].upper() == "MAJOR":
            idx = RomanToInt(chord) - 1
            if idx == -1:
                print("Wrong input format.")
                exit(-1)
            return key_map[major_key_Chordtype[idx]], isSeven
        else:
            idx = RomanToInt(chord) - 1
            if idx == -1:
                print("Wrong input format.")
                exit(-1)
            return key_map[minor_key_Chordtype[idx]], isSeven


def startPosition(key, chord, ctype, isSeven):
    isMajor, isFlat = False, False
    if key[-5:].upper() == "MAJOR":
        isMajor = True
    if chord[0].upper() == "B":
        isFlat = True
        chord = chord[1:]
    key = key[:-5]
    start_pos = pitch_to_index[key[0].upper()]
    if isSeven:
        chord = chord[:-1]
    if chord[-1:] == "+":
        chord = chord[:-1]
    if chord[-1:] == "-":
        chord = chord[:-1]
    if len(key) >= 2 and key[1].upper() == "B":
        start_pos -= 1
    elif len(key) >= 2 and key[1] == "#":
        start_pos += 1
    if ctype == "ger" or ctype == "fre" or ctype == "ita":
        start_pos -= 4
    elif ctype == "aug" and isMajor:
        start_pos += major_root_offset[RomanToInt(chord[3:]) - 1]
    elif ctype == "aug" and not isMajor:
        start_pos += minor_root_offset[RomanToInt(chord[3:]) - 1]
    elif ctype == "dim":
        if chord[:3].upper() == "DIM":
            chord = chord[3:]
        if isMajor:
            start_pos += major_root_offset[RomanToInt(chord) - 1]
        else:
            start_pos += minor_root_offset[RomanToInt(chord) - 1]
        if not isMajor and RomanToInt(chord) == 7:
            start_pos += 1
    elif isMajor:
        start_pos += major_root_offset[RomanToInt(chord) - 1]
    else:
        start_pos += minor_root_offset[RomanToInt(chord) - 1]
    if isFlat:
        start_pos -= 1
    if start_pos < 0:
        start_pos += 12
    if start_pos >= 12:
        start_pos -= 12
    return start_pos, isMajor


def noteNaming(notes, key, chord, ctype):
    output_notes = []
    # if ctype in ["ger","ita","fre"]: #These chords are same regardless of major or minor, therefore use notation of major
    #     key = key[:-5] + "Major"
    isMinor = False
    isEbMinor = False
    key = key.upper()
    if key[-5:] == "MINOR":
        if key == "EBMINOR":
            isEbMinor = True
        key = relative_major[key[:-5]] + "MAJOR"
        isMinor = True
    if key in [
        "CMAJOR",
        "GMAJOR",
        "DMAJOR",
        "AMAJOR",
        "EMAJOR",
        "BMAJOR",
        "F#MAJOR",
    ]:
        isFlat = False
    else:
        isFlat = True
    if isFlat:
        output_notes = [index_to_pitch_flat[y] for y in notes]
    else:
        output_notes = [index_to_pitch_sharp[y] for y in notes]
    if ctype in ["ger", "ita", "fre"]:
        if isFlat:
            output_notes[0] = index_to_pitch_doubleflat[notes[0]]  # 'b6'
            output_notes[-1] = index_to_pitch_sharp[notes[-1]]  # '#4'
        else:
            output_notes[0] = index_to_pitch_flat[notes[0]]  # 'b6'
            output_notes[-1] = index_to_pitch_doublesharp[notes[-1]]  # '#4'
        if ctype == "ger" and isFlat:
            output_notes[2] = index_to_pitch_doubleflat[notes[2]]  # 'b3'
        elif ctype == "ger" and not isFlat:
            output_notes[2] = index_to_pitch_flat[notes[2]]  # 'b3'
    if chord[0].upper() == "B":
        if isFlat:
            output_notes[0] = index_to_pitch_doubleflat[notes[0]]  # 'b2 or b6'
            output_notes[2] = index_to_pitch_doubleflat[notes[2]]  # 'b6 or b3'
        else:
            output_notes[0] = index_to_pitch_flat[notes[0]]  # 'b2 or b6'
            output_notes[2] = index_to_pitch_flat[notes[2]]  # 'b6 or b3'
    if len(chord) > 3 and chord[:3].upper() == "DIM" and len(notes) == 4:
        if isFlat:
            output_notes[3] = index_to_pitch_doubleflat[notes[3]]  # 'b6'
        else:
            output_notes[3] = index_to_pitch_flat[notes[3]]  # 'b6'
    if len(chord) > 3 and chord[:3].upper() == "DIM" and isMinor:
        if isFlat:
            output_notes[0] = index_to_pitch_sharp[notes[0]]
        else:
            output_notes[0] = index_to_pitch_doublesharp[notes[0]]
    if ctype == "aug":
        if isFlat:
            output_notes[2] = index_to_pitch_sharp[notes[2]]  # '#5'
            if len(notes) == 4:
                output_notes[3] = index_to_pitch_doubleflat[notes[3]]  # 'b7'
        else:
            output_notes[2] = index_to_pitch_doublesharp[notes[2]]  # '#5'
            if len(notes) == 4:
                output_notes[3] = index_to_pitch_flat[notes[3]]  # 'b7'
    if ctype == "maj" and len(chord) >= 2 and (chord[-1] == "+" or chord[-2] == "+"):
        if isFlat:
            output_notes[1] = index_to_pitch_sharp[notes[1]]
        else:
            output_notes[1] = index_to_pitch_doublesharp[notes[1]]
    if isEbMinor:
        for i, note in enumerate(output_notes):
            output_notes[i] = note if note != "B" else "Cb"
    return output_notes


def ChordToNote(key, chord):
    ctype, isSeven = typeAnalysis(key, chord)
    start, isMajor = startPosition(key, chord, ctype, isSeven)
    notes = []
    if ctype == "ger":
        for offset in german_offset:
            notes.append(start + offset)
    elif ctype == "fre":
        for offset in french_offset:
            notes.append(start + offset)
    elif ctype == "ita":
        for offset in italian_offset:
            notes.append(start + offset)
    elif ctype == "maj":
        if isSeven:
            for offset in major_offset:
                notes.append(start + offset)
            if isMajor:
                notes[3] += major_seventh_offset[RomanToInt(chord) - 1]
            else:
                notes[3] += minor_seventh_offset[RomanToInt(chord) - 1]
        else:
            for i in range(3):
                notes.append(start + major_offset[i])
    elif ctype == "min":
        if isSeven:
            for offset in minor_offset:
                notes.append(start + offset)
            if isMajor:
                notes[3] += major_seventh_offset[RomanToInt(chord) - 1]
            else:
                notes[3] += minor_seventh_offset[RomanToInt(chord) - 1]
        else:
            for i in range(3):
                notes.append(start + minor_offset[i])
    elif ctype == "dim":
        if isSeven:
            for offset in diminished_offset:
                notes.append(start + offset)
            if len(chord) < 3 or chord[:3].upper() != "DIM":
                if isMajor:
                    notes[3] += major_seventh_offset[RomanToInt(chord) - 1]
                else:
                    notes[3] += minor_seventh_offset[RomanToInt(chord) - 1]
        else:
            for i in range(3):
                notes.append(start + diminished_offset[i])
    else:
        if isSeven:
            for offset in augmented_offset:
                notes.append(start + offset)
        else:
            for i in range(3):
                notes.append(start + augmented_offset[i])
    notes = [x % 12 for x in notes]
    x = noteNaming(notes, key, chord, ctype)
    # print(f"The notes in {key}, {chord} chord are: {x}.")
    return notes, x  # change to return notes,x when it using Checker.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Output notes in a given chord.")
    parser.add_argument(
        "key",
        help="The targeted key (Format: Letter + major/minor without space, e.g. Cmajor)",
    )
    parser.add_argument(
        "chord",
        help="The targeted chord (Format: [Aug/Dim/Fre/Ger/Ita/b]Roman numeral[+/-/7] e.g. DimVII7)",
    )
    args = parser.parse_args()
    ChordToNote(args.key, args.chord)

