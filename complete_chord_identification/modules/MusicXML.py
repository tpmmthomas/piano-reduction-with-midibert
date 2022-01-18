from music21 import *

def read_musicXML(path):
    return converter.parse(path)

def get_measure_count(score):
    return len(score.parts[0].getElementsByClass('Measure'))

def get_key_signature(score):
    return score.parts[0].flat.getElementsByClass('KeySignature')

# Extract notes from music21's object
def get_notes(m21_object):
    if m21_object.duration.quarterLength <= 0.0001 or m21_object.isRest:
        return []
    if m21_object.isChord:
        for note in m21_object:
            note.offset = m21_object.offset
        m21_object[0].lyric = m21_object.lyric
        return [note for note in m21_object]
    else:
        return [m21_object]

# Extract all notes from measures of score
# With .flat means notes will be sorted according to their offset
def extract_notes_in_measures(score, start_measure_num, end_measure_num):
    all_notes = []
    measures = score.measures(start_measure_num, end_measure_num).flat
    for note in measures.flat.notes:
        all_notes += get_notes(note)
    return all_notes
'''
def extract_notes_in_measures(score, start_measure_num, end_measure_num):
    all_notes = []
    measure = score.measure(start_measure_num, end_measure_num)
    for part in measure.parts:
        components = [x for x in part.recurse().getElementsByClass('GeneralNote')]
        for x in components:
            all_notes += get_notes(x)
    return all_notes
'''
def get_notes_list(score):
    notes_list = []
    measure_num = get_measure_count(score)
    for i in range(0, measure_num + 1):
        measure = score.measure(i)
        temp = []
        for part in measure.parts:
            components = [x for x in part.recurse().getElementsByClass('GeneralNote')]
            for x in components:
                temp += get_notes(x)
        notes_list.append(temp)
    return notes_list