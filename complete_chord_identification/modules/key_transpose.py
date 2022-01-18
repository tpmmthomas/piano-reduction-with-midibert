from music21 import *
import music21
import os
import glob
import re
import numpy as np
import json


def key_transpose(piece, output):
    # key transpose profile
    MtoC = {
        "Gb": -6,
        "F": -5,
        "E": -4,
        "D#": -3,
        "Eb": -3,
        "D": -2,
        "C#": -1,
        "Db": -1,
        "C": 0,
        "B": 1,
        "A#": 2,
        "Bb": 2,
        "A": 3,
        "G#": 4,
        "Ab": 4,
        "G": 5,
        "F#": 6,
    }
    mtoA = {
        "D#": -6,
        "D": -5,
        "C#": -4,
        "Db": -4,
        "C": -3,
        "B": -2,
        "A#": -1,
        "Bb": -1,
        "A": 0,
        "G#": 1,
        "Ab": 1,
        "G": 2,
        "F#": 3,
        "Gb": 3,
        "F": 4,
        "E": 5,
        "Eb": 6,
    }
    threshold = 8  # in offset
    c = converter.parse(piece)
    for e in c.recurse().getElementsByClass(key.KeySignature):
        while e.sharps != 0:
            e.transpose(-1, inPlace=True)

    def cal_offset(e):
        if e is None:
            return 0
        return e.offset + cal_offset(e.activeSite)

    all_notes = []
    for el in c.recurse().notes:
        if el.lyric is not None:
            el.lyric = el.lyric.replace("♭", "b")
        all_notes.append([el.lyric, el, cal_offset(el)])
    b = sorted(all_notes, key=lambda x: x[0] if x[0] is not None else "ZZZ")
    b = sorted(b, key=lambda x: x[-1])
    k = None
    previous_change_offset = 0
    previous_change_index = 0
    for idx, e in enumerate(reversed(b)):
        if e[0] is None or "(" not in e[0]:
            # label without key
            e.append(0)
        else:
            # label with key
            if k is None:
                # initial condition
                k = e[0].split("(")[0]
                e.append(b[-1][2] - e[2])
                previous_change_offset = e[2]
                previous_change_index = idx
            else:
                # changed key
                if k != e[0].split("(")[0]:
                    e.append(previous_change_offset - e[2])
                    previous_change_offset = e[2]
                    previous_change_index = idx
                    k = e[0].split("(")[0]
                # no key change
                else:
                    # trace back
                    e.append(
                        previous_change_offset
                        - e[2]
                        + b[-previous_change_index - 1][-1]
                    )
                    b[-previous_change_index - 1][-1] = 0
                    previous_change_offset = e[2]
                    previous_change_index = idx
    current_key = None
    Major = None
    for e in b:
        if current_key is None:
            # fail-safe (idk wt is this for, seems just some empty meaningless thing ....)
            if "NoChord" in e[1].classes:
                continue

            if e[1].lyric is None:
                # infer key
                for i in b:
                    if i[0] is not None:
                        assert "(" in i[0]
                        # e[1].addLyric('infer '+i[0].split('(')[0])
                        current_key = i[0]
                        break
                else:
                    assert 1 == 0  # no key marked???
            else:
                current_key = e[1].lyric

            print("start")
            # e[1].addLyric('start')#every score should veriyf its start***

            # identify major or minor
            if "M" in current_key:
                Major = True
            else:
                Major = False

            # identify key name
            if Major:
                # e[1].addLyric('trans->CM')
                current_key = current_key.split("M")[0]
            else:
                # e[1].addLyric('trans->Am')
                current_key = current_key.split("m")[0]

        # check if key change
        elif (
            e[1].lyric is not None
            and ("m" in e[1].lyric or "M" in e[1].lyric)
            and "(" in e[1].lyric
        ):  # '(' --> is just to make sure it is key change

            # DEBUG
            # print(current_key,Major)
            # print(e[1].lyric)
            # print('----------------------')

            # check if key is different from previous key
            if (
                Major and "M" in e[1].lyric and current_key == e[1].lyric.split("M")[0]
            ) or (
                not Major
                and "m" in e[1].lyric
                and current_key == e[1].lyric.split("m")[0]
            ):
                pass
                # e[1].addLyric("same key as previous change")
            else:
                # update key
                if e[-1] < threshold:
                    pass
                    # e[1].addLyric("2nd dominant")
                else:
                    print("change key")
                    current_key = e[1].lyric

                    # identify major or minor
                    if "M" in current_key:
                        Major = True
                    else:
                        Major = False

                    # identify key name
                    if Major:
                        current_key = current_key.split("M")[0]
                        # e[1].addLyric("trans->CM")
                    else:
                        current_key = current_key.split("m")[0]
                        # e[1].addLyric("trans->Am")

        if Major:
            e[1].transpose(MtoC[current_key], inPlace=True)
        else:
            e[1].transpose(mtoA[current_key], inPlace=True)
    GEX = musicxml.m21ToXml.GeneralObjectExporter(c)
    out = GEX.parse()
    with open(output, "wb") as f:
        f.write(out)


import glob

for piece in glob.glob("../musicxml(ok)/*.mxl"):
    piecename = piece.split("(ok)\\")[-1]
    piecename = piecename[:-3] + "musicxml"
    print(piecename)

    print("../musicxml(transposed)/" + piecename)
    try:
        key_transpose(piece, "../musicxml(transposed)/trans_" + piecename)
    except:
        print("Error in ", piece)
print("Done!")
