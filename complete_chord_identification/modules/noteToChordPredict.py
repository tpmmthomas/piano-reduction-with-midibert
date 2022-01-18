from noteToChordWeighted import NoteToChord
import json
import numpy as np

with open("../data/training_data2.json", "r") as f:
    data = json.load(f)

##PROVIDED KEY VERSION
total_accuracy = []
for piece in list(data.keys()):
    # print(piece)
    correct = 0
    total = 0
    chord_seq = data[piece]["chord_seq"]
    note_seq = data[piece]["note_seq"]
    for i in range(len(chord_seq)):
        correctbool = False
        idx = chord_seq[i].find("_")
        key = chord_seq[i][:idx]
        key = key[:-1] + "Major" if key[-1] == "M" else key[:-1] + "Minor"
        chord = chord_seq[i][idx + 1 :]
        # print(note_seq[i], key)
        result = NoteToChord(note_seq[i], key, numOut=1)
        if not result is None:
            result = result[0]["Chord"]
        else:
            # print(f"Expected: {chord}, returned: {result}, Correct: {correctbool}")
            # print("For checking:", chord_seq[i], note_seq[i])
            total += 1
            continue
        # print(key, chord, result, note_seq[i])
        idx = result.find("or")
        result = result[idx + 2 :]
        # print(chord, result)
        # print(f"Expected: {chord}, returned: {result}")
        if chord == "Dim7":
            chord = "VII"
        if result.find("VII") != -1:
            result = "VII"
        if (
            len(chord) >= 3
            and len(result) >= 3
            and chord[:3].upper() == result[:3].upper()
        ):
            correct += 1
            correctbool = True
        else:
            if chord[-1] == "7":
                chord = chord[:-1]
            if result[-1] == "7":
                result = result[:-1]
            if chord == result:
                correct += 1
                correctbool = True
        total += 1
        # print(f"Expected: {chord}, returned: {result}, Correct: {correctbool}")
    print(piece, "Accuracy: ", round(correct / total * 100, 2), "%")
    total_accuracy.append(round(correct / total * 100, 2))

print("Statistics:")
print("Max: ", np.max(total_accuracy))
print("Min: ", np.min(total_accuracy))
print("Average: ", np.mean(total_accuracy))
