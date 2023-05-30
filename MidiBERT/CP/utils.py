import numpy as np
import miditoolkit

# parameters for input
DEFAULT_VELOCITY_BINS = np.array(
    [0, 32, 48, 64, 80, 96, 128]
)  # np.linspace(0, 128, 32+1, dtype=np.int)

DEFAULT_BEATS = 4

# For 24, 34, 44
# Each bar -> 24, 36, 48 beats
# Each beat -> 12 beats (always the same)
DEFAULT_FRACTION = 48
DEFAULT_FRACTION_PER_BEAT = DEFAULT_FRACTION / DEFAULT_BEATS

DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_SUB_TICKS_PER_BEAT = 480 / DEFAULT_FRACTION_PER_BEAT
DEFAULT_TICKS_PER_BAR = DEFAULT_TICKS_PER_BEAT * DEFAULT_BEATS

# Duration MAX = across 4 bars
# [40, 80, 120, 240, ..., ]
DEFAULT_MAX_BAR_DURATION = 4
DEFAULT_DURATION_BINS = np.arange(
    DEFAULT_SUB_TICKS_PER_BEAT,
    DEFAULT_SUB_TICKS_PER_BEAT
    * DEFAULT_FRACTION_PER_BEAT
    * DEFAULT_BEATS
    * DEFAULT_MAX_BAR_DURATION
    + 1,
    DEFAULT_SUB_TICKS_PER_BEAT,
    dtype=int,
)

# parameters for output
DEFAULT_RESOLUTION = DEFAULT_TICKS_PER_BEAT

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
        self.Program = -1
        self.TimeSignature = "00"

    def __repr__(self):
        return "Item(name={}, start={}, end={}, velocity={}, pitch={}, Type={}, Program={}, Time Signature={})".format(
            self.name,
            self.start,
            self.end,
            self.velocity,
            self.pitch,
            self.Type,
            self.Program,
            self.TimeSignature,
        )


# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path, is_reduction=False):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)

    # Filtering Rules
    # 1. If the midi contains incompatible time signature
    for ts in midi_obj.time_signature_changes:
        if (ts.numerator not in [2, 3, 4]) or ts.denominator != 4:
            return [], []
    # 2. If the number of valid tracks is less than half
    total, valid = 0, 0
    for inst in midi_obj.instruments:
        if inst.program < 96:
            valid += 1
        total += 1
    if valid * 2 <= total or total < 3:
        return [], []

    # note
    note_items = []
    num_of_instr = len(midi_obj.instruments)
    tpbo = midi_obj.ticks_per_beat

    for i in range(num_of_instr):
        if midi_obj.instruments[i].is_drum or midi_obj.instruments[i].program >= 97:
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

    note_items.sort(
        key=lambda x: (x.start, x.pitch)
    )  # This is how the notes should be ordered

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


def item2event(groups, task, numerator=4, midi_obj=None):
    events = []
    n_downbeat = 0
    current_time = 0
    for i in range(len(groups)):
        if midi_obj != None:
            ts = raw_time_signature(midi_obj, current_time)
            numerator = int(ts[0])

        if "Note" not in [item.name for item in groups[i][1:-1]]:
            note_tuple = ABS_event(ts)
            events.append(note_tuple)
            n_downbeat += 1
            current_time += DEFAULT_TICKS_PER_BEAT * numerator
            continue

        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        current_time += DEFAULT_TICKS_PER_BEAT * numerator
        new_bar = True

        for item in groups[i][1:-1]:
            # Handle notes only
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

            # ====================================================================
            # Position
            flags = np.linspace(
                bar_st, bar_et, int(DEFAULT_FRACTION / 4 * numerator), endpoint=False
            )
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
            # ====================================================================

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

            # ====================================================================
            if task == "custom" or task == "skyline":
                # Program
                note_tuple.append(
                    Event(
                        name="Program",
                        time=item.start,
                        value=item.Program,
                        text="{}".format(item.Program),
                        Type=-1,
                    )
                )

                # Time Signature
                note_tuple.append(
                    Event(
                        name="Time Signature",
                        time=item.start,
                        value=item.TimeSignature,
                        text="{}".format(item.TimeSignature),
                        Type=-1,
                    )
                )
            # ====================================================================

            events.append(note_tuple)
    return events


def quantize_items(items, ticks=DEFAULT_SUB_TICKS_PER_BEAT):
    grids = np.arange(0, items[-1].start + 1, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
        item.shift = shift
    return items


def group_items(
    items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4, multiple_ts_at=[0]
):
    items.sort(key=lambda x: x.start)

    if multiple_ts_at[0] != 0:
        multiple_ts_at[0] = 0
    while items[-1].start < multiple_ts_at[-1]:
        multiple_ts_at = multiple_ts_at[:-1]

    if len(multiple_ts_at) == 1:
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

    if len(multiple_ts_at) > 1:
        ts_index = 0
        gp = []
        gps_buf = []
        for i in range(len(items)):
            if ts_index + 1 >= len(multiple_ts_at):
                gp += items[i:]
                gps_buf += [gp]
                gp = []
                break
            if (
                multiple_ts_at[ts_index]
                <= items[i].start
                < multiple_ts_at[ts_index + 1]
            ):
                gp += [items[i]]
            else:
                gps_buf += [gp]
                gp = [items[i]]
                ts_index += 1
        if len(gp) > 0:
            gps_buf += [gp]

    groups = []
    for idx, ts_time in enumerate(multiple_ts_at):
        numerator = int(gps_buf[idx][0].TimeSignature[0])
        ticks_per_bar = DEFAULT_RESOLUTION * numerator
        if idx + 1 >= len(multiple_ts_at):
            stop_time = gps_buf[idx][-1].start + ticks_per_bar + 1
        else:
            stop_time = multiple_ts_at[idx + 1] + ticks_per_bar

        downbeats = np.arange(multiple_ts_at[idx], stop_time, ticks_per_bar)
        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            insiders = []
            for item in items:
                if (item.start >= db1) and (item.start < db2):
                    insiders.append(item)
            overall = [db1] + insiders + [db2]
            groups.append(overall)
    return groups


def convert_string_quartets(midi_obj):
    # Convert eligible programs [40, 40, 42, 42] to [45, 40, 41, 42]
    programs = [instrument.program for instrument in midi_obj.instruments]
    if programs == [40, 40, 42, 42] and "viola" in midi_obj.instruments[2].name:
        midi_obj.instruments[0].program = 45
        midi_obj.instruments[2].program = 41

    # Convert programs [40, 40, 41, 42] to [45, 40, 41, 42]
    if programs == [40, 40, 41, 42]:
        midi_obj.instruments[0].program = 45
    return midi_obj

def is_string_quartets(midi_obj):
    programs = [instrument.program for instrument in midi_obj.instruments]
    return programs == [45, 40, 41, 42]


# ====================================================================
def Type2Program(midi_obj, channel):
    """Get Program Change Number from the instrument name at a specified channel

    Parameters:
        midi_obj (MidiFile): from miditoolkit.midi.parser.MidiFile
        channel (int): the index of the channel, 0-based

    Returns:
        int: Program Change Number, 0-based

    """
    instrument = midi_obj.instruments[channel]
    program_number = instrument.program
    return program_number


def raw_time_signature(midi_obj, time):
    """Get the Time Signature at the specified time

    Parameters:
        midi_obj (obj): miditoolkit.midi.parser.MidiFile
        time (int): the specified time

    Returns:
        str: the Time Signature at that time, e.g. "44", "34"

    """
    time_signature_changes = [i.time for i in midi_obj.time_signature_changes]
    # Scale Time Signature
    tpbo = midi_obj.ticks_per_beat
    time_signature_changes = [
        i / tpbo * DEFAULT_RESOLUTION for i in time_signature_changes
    ]
    # Get the range by index
    idx = np.digitize(time, time_signature_changes) - 1
    # The specified time should be after/at the first note
    if idx < 0:
        idx = 0
    numerator = midi_obj.time_signature_changes[idx].numerator
    denominator = midi_obj.time_signature_changes[idx].denominator
    return f"{numerator}{denominator}"


def ABS_event(ts="44"):
    note_tuple = []
    note_tuple.append(
        Event(
            name="Bar",
            time=None,
            value="<ABS>",
            text="<ABS>",
            Type=-1,
        )
    )
    note_tuple.append(
        Event(
            name="Position",
            time=None,
            value="<ABS>",
            text="<ABS>",
            Type=-1,
        )
    )
    note_tuple.append(
        Event(
            name="Pitch",
            time=None,
            value="<ABS>",
            text="<ABS>",
            Type=-1,
        )
    )
    note_tuple.append(
        Event(
            name="Duration",
            time=None,
            value="<ABS>",
            text="<ABS>",
            Type=-1,
        )
    )
    note_tuple.append(
        Event(
            name="Program",
            time=None,
            value="<ABS>",
            text="<ABS>",
            Type=-1,
        )
    )
    note_tuple.append(
        Event(
            name="Time Signature",
            time=None,
            value=ts,
            text=ts,
            Type=-1,
        )
    )
    return note_tuple


# ====================================================================
