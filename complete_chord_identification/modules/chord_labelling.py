from music21 import *
import music21
import os
import glob
import re
import numpy as np
from noteToChordWeighted import NoteToChord
import json

chord_sequence = {}
i = 0


def demonopolize(notelist):
    if len(notelist) <= 2:
        return notelist
    other_values = np.array(list(notelist.values()))
    other_values = other_values[other_values < 0.5]
    maxval = np.max(other_values)
    if len(notelist) <= 2:
        return notelist
    hasChanged = True
    dontchange = []
    totalredist = 1
    for note in notelist:
        if notelist[note] >= 0.5:
            notelist[note] = maxval
    return notelist


def importance_score(notelist, noteduration, noteoctave):
    allnotes = {}
    for i in range(len(notelist)):
        if not notelist[i] in allnotes:
            allnotes[notelist[i]] = {
                "occ": 1,
                "durlist": [noteduration[i]],
                "octavelist": [noteoctave[i]],
            }
        else:
            allnotes[notelist[i]]["occ"] += 1
            allnotes[notelist[i]]["durlist"].append(noteduration[i])
            allnotes[notelist[i]]["octavelist"].append(noteoctave[i])
    returnnote = {}
    totalscore = 0
    for note in allnotes:
        returnnote[note] = int(
            allnotes[note]["occ"]
            * (np.sum(allnotes[note]["durlist"]))
            * (21 - 2 * np.min(allnotes[note]["octavelist"]))
        )
        if returnnote[note] == 0:
            returnnote[note] = 1
        totalscore += returnnote[note]
    for note in allnotes:
        returnnote[note] = round(returnnote[note] / totalscore, 3)
    returnnote = demonopolize(returnnote)
    return returnnote


for piece in glob.glob("../musicxml(not_notated)/*.musicxml"):
    #     if i == 0:
    #         i += 1
    #         continue
    print(piece)
    chords = []
    notes = []
    c = converter.parse(piece)
    partStream = c.parts.stream()
    post = c.flattenParts().flat
    allText = text.assembleLyrics(post)
    print(allText)
    if len(allText) == 0:
        continue
    #     firstChord = False
    current_key = ""
    current_chord = ""
    notelist = []
    noteoffset = []
    noteduration = []
    noteoctave = []
    for note in post.notes:
        if not note.lyric is None:
            newchord = note.lyric
            newnotelist = []
            newnoteoffset = []
            newnoteduration = []
            newnoteoctave = []
            if current_key != "" and current_chord != "" and len(notelist) != 0:
                chords.append(current_chord.replace("\u266d", "b"))
                toremove = []
                for i, offset in enumerate(noteoffset):
                    if offset == note.offset:
                        newnotelist.append(notelist[i])
                        newnoteoffset.append(noteoffset[i])
                        newnoteduration.append(noteduration[i])
                        newnoteoctave.append(noteoctave[i])
                        toremove.append(i)
                for idx in reversed(toremove):
                    notelist.pop(idx)
                    noteoffset.pop(idx)
                    noteduration.pop(idx)
                    noteoctave.pop(idx)
                append_list = importance_score(notelist, noteduration, noteoctave)
                notes.append(append_list)
            notelist = newnotelist
            noteoffset = newnoteoffset
            noteduration = newnoteduration
            noteoctave = newnoteoctave
            chordidx = newchord.find("(")
            if chordidx != -1:
                newkey = newchord[0:chordidx]
                newchord = newchord[chordidx + 1 : -1]
                current_key = newkey
            current_chord = current_key + "_" + newchord
        allnotes = list(note.pitches)
        duration = note.duration
        for note1 in allnotes:
            notelist.append(note1.name.replace("\u266d", "b"))
            noteoffset.append(note.offset)
            noteduration.append(duration.quarterLength)
            noteoctave.append(int(note1.nameWithOctave[-1]))
    if current_key != "" and current_chord != "" and len(notelist) != 0:
        chords.append(current_chord.replace("\u266d", "b"))
        append_list = importance_score(notelist, noteduration, noteoctave)
        notes.append(append_list)
    piecekey = re.sub(r"[^a-zA-Z0-9_.]", "", piece[23:])
    chord_sequence[piecekey] = {"chord_seq": chords, "note_seq": notes}

# Then call note to chord
# Then yub
print(chord_sequence)

predictions = {}

for piecekey in chord_sequence:
    predictions[piecekey] = []
    currentpiece = chord_sequence[piecekey]
    piecechord = currentpiece["chord_seq"]
    piecenote = currentpiece["note_seq"]
    for i in range(len(piecechord)):
        idx = piecechord[i].find("_")
        key = piecechord[i][:idx]
        key = key[:-1] + "Major" if key[-1] == "M" else key[:-1] + "Minor"
        result = NoteToChord(piecenote[i], key, numOut=1)
        if not result is None:
            hasSeventh = result[0]["hasSeventh"]
            result = result[0]["Chord"]
            if hasSeventh:
                chordidx = result.find("or")
                chordx = result[chordidx + 2 :]
                if result.find("Major") != -1 and chordx in [
                    "II",
                    "V",
                    "VII",
                    "DimVII",
                ]:
                    result += "7"
                elif result.find("Minor") != -1 and chordx in [
                    "II",
                    "V",
                    "V+",
                    "DimVII",
                ]:
                    result += "7"
        else:
            result = "None"
        predictions[piecekey].append(result)

for piece in glob.glob("../musicxml(not_notated)/*.musicxml"):
    piecekey = re.sub(r"[^a-zA-Z0-9_.]", "", piece[23:])
    print(piecekey)
    predict = predictions[piecekey]
    i = 0
    c = converter.parse(piece)
    post = c.flattenParts().flat
    for thisNote in post.notes:
        if thisNote.lyric is not None:
            if thisNote.lyric == "x":
                j = predict[i].find("or")
                toadd = predict[i][j + 2 :]
                thisNote.addLyric(toadd)
            else:
                q = thisNote.lyric.find("(")
                j = predict[i].find("or")
                toadd = thisNote.lyric[:q] + "(" + predict[i][j + 2 :] + ")"
                thisNote.addLyric(toadd)
            print(toadd)
            i += 1
    GEX = musicxml.m21ToXml.GeneralObjectExporter(post)
    out = GEX.parse()
    # outstr = out.decode("utf-8")
    # print(outstr.strip())
    with open(piece[:-9] + "_n2caddit.musicxml", "wb") as f:
        f.write(out)

