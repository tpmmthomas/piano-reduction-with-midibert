import numpy as np
import miditoolkit
import copy
import os

# parameters for input
DEFAULT_VELOCITY_BINS = np.array(
    [0, 32, 48, 64, 80, 96, 128]
)  # np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch, Type, shift=0):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.Type = Type
        self.shift = shift

    def __repr__(self):
        return "Item(name={}, start={}, end={}, velocity={}, pitch={}, Type={})".format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.Type
        )


def read_midi(path):
    mido_obj = miditoolkit.midi.parser.MidiFile(path)
    tick_per_beat = mido_obj.ticks_per_beat

    notes = []
    for instrument in mido_obj.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            # rescale tpb to 480
            note.start = int(note.start / tick_per_beat * DEFAULT_RESOLUTION)
            note.end = int(note.end / tick_per_beat * DEFAULT_RESOLUTION)
            notes.append(note)

    # sort by start time
    notes.sort(key=lambda note: note.start)
    return notes, 480


def mergeIntervals(arr):
    # Sorting based on the increasing order
    # of the start intervals
    arr.sort(key=lambda x: x[0])
    # array to hold the merged intervals
    m = []
    s = -10000
    max = -100000
    for i in range(len(arr)):
        a = arr[i]
        if a[0] > max:
            if i != 0:
                m.append([s, max])
            max = a[1]
            s = a[0]
        else:
            if a[1] >= max:
                max = a[1]
    #'max' value gives the last point of
    # that particular interval
    # 's' gives the starting point of that interval
    # 'm' array contains the list of all merged intervals
    if max != -100000 and [s, max] not in m:
        m.append([s, max])
    return m


def interval_histogram(notep):
    hist_p = dict()
    for note in notep:
        if note.pitch in hist_p:
            hist_p[note.pitch].append([note.start, note.end])
            hist_p[note.pitch] = mergeIntervals(hist_p[note.pitch])
        else:
            hist_p[note.pitch] = [[note.start, note.end]]
    return hist_p


# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path, is_reduction=False):
    if is_reduction:
        midi_obj = miditoolkit.midi.parser.MidiFile(
            os.path.join(file_path, "orchestra.mid")
        )

    else:
        midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    num_of_instr = len(midi_obj.instruments)
    tpbo = midi_obj.ticks_per_beat
    time_signatures = midi_obj.time_signature_changes

    for ts in time_signatures:
        if not (ts.numerator == 4 and ts.denominator == 4):
            return [], []

    for i in range(num_of_instr):
        if midi_obj.instruments[i].is_drum:
            continue
        notes = midi_obj.instruments[i].notes
        notes.sort(key=lambda x: (x.start, x.pitch))

        for note in notes:
            if note.pitch < 22 or note.pitch > 107:
                continue
            note_items.append(
                Item(
                    name="Note",
                    start=int(note.start / tpbo * DEFAULT_RESOLUTION),
                    end=int(note.end / tpbo * DEFAULT_RESOLUTION),
                    velocity=note.velocity,
                    pitch=note.pitch,
                    Type=i,
                )
            )

    note_items.sort(key=lambda x: (x.start, x.pitch))

    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(
            Item(
                name="Tempo",
                start=tempo.time,
                end=None,
                velocity=None,
                pitch=int(tempo.tempo),
                Type=-1,
            )
        )
    tempo_items.sort(key=lambda x: x.start)

    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(
                Item(
                    name="Tempo",
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=existing_ticks[tick],
                    Type=-1,
                )
            )
        else:
            output.append(
                Item(
                    name="Tempo",
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=output[-1].pitch,
                    Type=-1,
                )
            )
    tempo_items = output

    if is_reduction:
        notep, tpbp = read_midi(os.path.join(file_path, "piano.mid"))
        # if tpbp != tpbo:
        #     print("GG tpb different")
        #     return note_items,tempo_items,None
        histp = interval_histogram(notep)
        return note_items, tempo_items, histp

    return note_items, tempo_items


class Event(object):
    def __init__(self, name, time, value, text, Type):
        self.name = name
        self.time = time
        self.value = value
        self.text = text
        self.Type = Type

    def __repr__(self):
        return "Event(name={}, time={}, value={}, text={}, Type={})".format(
            self.name, self.time, self.value, self.text, self.Type
        )


def item2event(groups, task):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if "Note" not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        new_bar = True

        for item in groups[i][1:-1]:
            if item.name != "Note":
                continue
            note_tuple = []

            # Bar
            if new_bar:
                BarValue = "New"
                new_bar = False
            else:
                BarValue = "Continue"
            note_tuple.append(
                Event(
                    name="Bar",
                    time=None,
                    value=BarValue,
                    text="{}".format(n_downbeat),
                    Type=-1,
                )
            )

            # Position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags - item.start))
            note_tuple.append(
                Event(
                    name="Position",
                    time=item.start,
                    value="{}/{}".format(index + 1, DEFAULT_FRACTION),
                    text="{}".format(item.start),
                    Type=-1,
                )
            )

            # Pitch
            velocity_index = (
                np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side="right") - 1
            )

            if task == "melody":
                pitchType = item.Type
            elif task == "velocity":
                pitchType = velocity_index
            else:
                pitchType = -1

            note_tuple.append(
                Event(
                    name="Pitch",
                    time=item.start,
                    value=item.pitch,
                    text="{}".format(item.pitch),
                    Type=pitchType,
                )
            )

            # Duration
            duration = item.end - item.start
            index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
            note_tuple.append(
                Event(
                    name="Duration",
                    time=item.start,
                    value=index,
                    text="{}/{}".format(duration, DEFAULT_DURATION_BINS[index]),
                    Type=-1,
                )
            )

            if task == "reduction":
                note_tuple.append(
                    Event(
                        name="genLabel",
                        time=item.start,
                        value=(item.start - item.shift, item.end - item.shift),
                        text="{},{}".format(item.start, item.end),
                        Type=-1,
                    )
                )
            events.append(note_tuple)

    return events


def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
        item.shift = shift
    return items


def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups
