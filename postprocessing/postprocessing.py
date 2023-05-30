#!/usr/bin/env python
# coding: utf-8

from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
from numpy import array, linspace
from sklearn.neighbors import KernelDensity
from matplotlib.pyplot import plot
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d
import numpy as np
from miditoolkit.pianoroll import parser as pr_parser
from miditoolkit.pianoroll import utils
import matplotlib.pyplot as plt
import math
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def read_midi(path, tpb=0):
    global DEFAULT_RESOLUTION
    mido_obj = mid_parser.MidiFile(path)
    tick_per_beat = mido_obj.ticks_per_beat
    if tpb == 0:
        DEFAULT_RESOLUTION = tick_per_beat
    else:
        DEFAULT_RESOLUTION = tpb

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
    return notes, DEFAULT_RESOLUTION


def write_midi(
    notes, centers_per_beat, path="out.mid", tick_per_beat=480, note_rh=None
):
    out = mid_parser.MidiFile()
    out.ticks_per_beat = tick_per_beat
    out.instruments = [
        ct.Instrument(program=0, is_drum=False, name="right hand"),
        ct.Instrument(program=0, is_drum=False, name="left hand"),
    ]
    if note_rh is not None:
        for note in notes:
            assert note.velocity
            out.instruments[1].notes.append(
                ct.Note(
                    start=note.start,
                    end=note.end,
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )
        for note in note_rh:
            assert note.velocity
            out.instruments[0].notes.append(
                ct.Note(
                    start=note.start,
                    end=note.end,
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
            )
    else:
        for note in notes:
            assert note.velocity
            current_beat = note.start // tick_per_beat
            # centers = centers_per_beat[current_beat]
            closest_center, abs_distance_to_centers = cluster_assignment(
                note, centers_per_beat, tick_per_beat, single_note=True
            )
            if closest_center == 0:
                out.instruments[1].notes.append(
                    ct.Note(
                        start=note.start,
                        end=note.end,
                        pitch=note.pitch,
                        velocity=note.velocity,
                    )
                )
            else:
                out.instruments[0].notes.append(
                    ct.Note(
                        start=note.start,
                        end=note.end,
                        pitch=note.pitch,
                        velocity=note.velocity,
                    )
                )
    out.dump(path)


# note that are holding in this beat
def get_notes_of_beat(notes, beat, tick_per_beat):
    start_tick = beat * tick_per_beat + 1
    end_tick = (beat + 1) * tick_per_beat - 1
    return list(
        filter(
            lambda n: n.start <= start_tick <= n.end
            or n.start <= end_tick <= n.end
            or start_tick <= n.start <= n.end <= end_tick,
            notes,
        )
    )


# mode: normal, centroid, minmax
def get_centers_per_beat(notes, tick_per_beat, kernel_size=5, mode="normal"):
    # find centers
    centers_per_beat = []
    range_per_beat = []

    length = math.ceil(max(notes, key=lambda x: x.end).end / tick_per_beat)
    # print(length,"debug")
    for beat in range(length):
        notes_within_window = get_notes_of_beat(notes, beat, tick_per_beat)
        pitches_within_window = np.array([note.pitch for note in notes_within_window])
        if len(pitches_within_window) == 0:
            range_per_beat.append(None)
            continue
        upper_limit, lower_limit = max(pitches_within_window), min(
            pitches_within_window
        )
        range_per_beat.append([lower_limit, upper_limit])

    pitch_range = linspace(0, 127)
    for beat in range(length):
        notes_within_window = get_notes_of_beat(notes, beat, tick_per_beat)
        pitches_within_window = np.array([note.pitch for note in notes_within_window])
        if len(pitches_within_window) == 0:
            centers_per_beat.append(None)
            continue

        # use KDE for clustering(intervaling)(determine the center line for left/right hand respectively)
        pitches_within_window = pitches_within_window.reshape(-1, 1)
        kernel_size = 6
        # interatve clustering
        while True:
            kde = KernelDensity(kernel="gaussian", bandwidth=kernel_size).fit(
                pitches_within_window
            )
            kde_val = kde.score_samples(pitch_range.reshape(-1, 1))
            # visualization: distribution of notes inside the window
            # plot(pitch_range, kde_val)
            minima, maxima = (
                argrelextrema(kde_val, np.less)[0],
                argrelextrema(kde_val, np.greater)[0],
            )

            # termination check
            if len(minima) >= 1 and len(maxima) >= 2:
                # cluster selection
                cutoff = min(pitch_range[minima])
                cluster_1_candidate = pitch_range[maxima[pitch_range[maxima] < cutoff]]
                cluster_2_candidate = pitch_range[maxima[pitch_range[maxima] > cutoff]]
                assert len(cluster_1_candidate) >= 1
                assert len(cluster_2_candidate) >= 1
                clusters = sorted(
                    [round(max(cluster_1_candidate)), round(max(cluster_2_candidate))]
                )

                centers_per_beat.append(clusters)
                break

            elif kernel_size < 0.000000000001:
                # probably only one hand is needed
                pitches_within_window.sort()
                if len(notes_within_window) >= 3:
                    centers_per_beat.append(
                        [
                            round(
                                np.average(
                                    pitches_within_window[
                                        : len(pitches_within_window) // 2
                                    ]
                                )
                            ),
                            round(
                                np.average(
                                    pitches_within_window[
                                        len(pitches_within_window) // 2 :
                                    ]
                                )
                            ),
                        ]
                    )
                else:
                    if round(np.average(pitches_within_window)) >= 60:
                        centers_per_beat.append(
                            [
                                round(np.average(pitches_within_window)) - 1,
                                round(np.average(pitches_within_window)),
                            ]
                        )
                    else:
                        centers_per_beat.append(
                            [
                                round(np.average(pitches_within_window)),
                                round(np.average(pitches_within_window)) + 1,
                            ]
                        )

                break

            else:
                kernel_size *= 0.7
                continue

        # diff modes
        if centers_per_beat[-1][0] > centers_per_beat[-1][1]:
            temp = centers_per_beat[-1][0]
            centers_per_beat[-1][0] = centers_per_beat[-1][1]
            centers_per_beat[-1][1] = temp
        centers = np.array(centers_per_beat[-1])
        left_candidate_notes = []
        right_candidate_notes = []

        for note in notes_within_window:
            abs_distance_to_centers = abs(note.pitch - centers)
            closest_center = np.argmin(abs_distance_to_centers)  # 0=left, 1 = right
            if closest_center == 0:
                left_candidate_notes.append(note)
            else:
                right_candidate_notes.append(note)

        # DEBUG
        left_notes_pitch = np.array([n.pitch for n in left_candidate_notes])
        right_notes_pitch = np.array([n.pitch for n in right_candidate_notes])
        #        print(left_candidate_notes,right_candidate_notes)

        if mode == "centroid":
            # left centroid
            if len(left_notes_pitch) > 0:
                left_centroid = np.argmin(abs(left_notes_pitch - centers[0]))
                centers_per_beat[-1][0] = left_notes_pitch[left_centroid]
            # right centroid
            if len(right_notes_pitch) > 0:
                right_centroid = np.argmin(abs(right_notes_pitch - centers[1]))
                centers_per_beat[-1][1] = right_notes_pitch[right_centroid]

        # print(centers_per_beat[-1],left_notes_pitch,right_notes_pitch,notes_within_window)

        elif mode == "minmax":
            if len(left_notes_pitch) > 0:
                left_centroid = np.argmin(left_notes_pitch)
                centers_per_beat[-1][0] = left_notes_pitch[left_centroid]
            if len(right_notes_pitch) > 0:
                right_centroid = np.argmax(right_notes_pitch)
                centers_per_beat[-1][1] = right_notes_pitch[right_centroid]

    # swap center
    for i, center in enumerate(centers_per_beat):
        if center is not None:
            if center[0] > center[1]:
                temp = centers_per_beat[i][0]
                centers_per_beat[i][0] = centers_per_beat[i][1]
                centers_per_beat[i][1] = temp

    # fill None
    for i, r in enumerate(range_per_beat):
        if r is None:
            left = None
            right = None
            for leftblk in reversed(range_per_beat[:i]):
                if leftblk is not None:
                    left = leftblk
                    break
            for rightblk in range_per_beat[i:]:
                if rightblk is not None:
                    right = rightblk
                    break
            if left is None:
                left = right
            if right is None:
                right = left
            assert (
                left is not None and right is not None
            )  # the range_per_beat cannot be None
            range_per_beat[i] = np.array(left) + np.array(right) // 2
    for i, r in enumerate(centers_per_beat):
        if r is None:
            left = None
            right = None
            for leftblk in reversed(centers_per_beat[:i]):
                if leftblk is not None:
                    left = leftblk
                    break
            for rightblk in centers_per_beat[i:]:
                if rightblk is not None:
                    right = rightblk
                    break
            if left is None:
                left = right
            if right is None:
                right = left
            assert (
                left is not None and right is not None
            )  # the centers_per_beat cannot be None
            centers_per_beat[i] = np.array(left) + np.array(right) // 2

    # smooth the line
    centers_per_beat = np.array(centers_per_beat)
    centers_per_beat[:, 0] = gaussian_filter1d(centers_per_beat[:, 0], kernel_size)
    centers_per_beat[:, 1] = gaussian_filter1d(centers_per_beat[:, 1], kernel_size)

    assert center_ordering(centers_per_beat)
    return np.array(centers_per_beat), np.array(range_per_beat)


def center_ordering(centers_per_beat):
    # make sure the cluster with lower octave stay at index 0 (left hand)
    for i, center in enumerate(centers_per_beat):
        if center is None:
            continue
        if center[0] > center[1]:
            # print(i,center)
            return False
    else:
        return True


def cluster_assignment(notes, centers_per_beat, tick_per_beat, single_note=False):
    def assignment(note, centers_per_beat):
        current_beat = note.start // tick_per_beat
        centers = centers_per_beat[current_beat]
        abs_distance_to_centers = abs(note.pitch - centers)
        closest_center = np.argmin(abs_distance_to_centers)
        return closest_center, abs_distance_to_centers

    assert center_ordering(centers_per_beat)
    if single_note == True:
        # return idx of cluster (0:left, 1:right)
        return assignment(notes, centers_per_beat)

    else:
        classified = [[], []]
        abs_distance_to_classified_centers = [[], []]
        for note in notes:
            closest_center, abs_distance_to_centers = assignment(note, centers_per_beat)
            classified[closest_center].append(note)
            abs_distance_to_classified_centers[closest_center].append(
                abs_distance_to_centers
            )
        return classified, abs_distance_to_classified_centers


# filter long duration notes
def trim_long_notes(notes, tick_per_beat):
    def trim(note):
        if note.end - note.start > threshold * tick_per_beat:
            note.end = note.start + threshold * tick_per_beat
        return note

    threshold = 4
    return list(map(trim, notes))


def octave_transpose(
    transpose_notes,
    tick_per_beat,
    threshold=10,
    center_threshold=5,
    minmaxrange=None,
    mode="normal",
    shiftbase=False,
    deletesparse=False,
):
    # octave transpose
    # for simplicity, use semitone instead of octave
    max_semitone_distance = threshold

    centers_per_beat, range_per_beat = get_centers_per_beat(
        transpose_notes, tick_per_beat, center_threshold, mode=mode
    )
    # use user provided range_per_beat
    if minmaxrange is not None:
        assert len(minmaxrange) >= len(range_per_beat)
        range_per_beat = minmaxrange

    transpose_track = []
    skip_track = []
    count = 0
    skip_count = 0

    # ---remove sparse note
    sparse_threshold = 14  # semitone
    delete_sparse = set()
    # ---end remove sparse note

    cached_note_beat = 0
    cached_left_notes = []
    cached_right_notes = []
    all_left_notes, all_right_notes = [], []
    shift_up_count = 0
    for note_idx, note in enumerate(transpose_notes):
        # cluster assignment
        current_beat = note.start // tick_per_beat
        centers = centers_per_beat[current_beat]
        closest_center, abs_distance_to_centers = cluster_assignment(
            note, centers_per_beat, tick_per_beat, single_note=True
        )

        # ---------------shift up(left hand)------------------------------
        # SHIFTBASE WITHOUT CHANGING THE NOTE ASSIGNMENT
        # print('---------------------------------------------')
        # print(note)
        # print(current_beat)
        # print(cached_left_notes)
        # print(cached_right_notes)
        if len(cached_left_notes) > 0 and len(cached_right_notes) > 0:
            # print('start shift up?')
            # assert all notes in the same beat
            assert all(
                [
                    n[1].start // tick_per_beat
                    == cached_left_notes[0][1].start // tick_per_beat
                    for n in cached_left_notes
                ]
            )
            assert all(
                [
                    n[1].start // tick_per_beat
                    == cached_right_notes[0][1].start // tick_per_beat
                    for n in cached_right_notes
                ]
            )
            assert (
                cached_left_notes[0][1].start // tick_per_beat
                == cached_right_notes[0][1].start // tick_per_beat
            )

            if current_beat > cached_left_notes[0][1].start // tick_per_beat:
                # print('condition passed')
                left_highest_pitch = max(cached_left_notes, key=lambda n: n[1].pitch)[
                    1
                ].pitch
                right_lowest_pitch = min(cached_right_notes, key=lambda n: n[1].pitch)[
                    1
                ].pitch
                # print(left_highest_pitch,right_lowest_pitch,right_lowest_pitch-left_highest_pitch)
                if shiftbase and right_lowest_pitch - left_highest_pitch > 18:
                    # print('shift up')
                    # print('shift up ',cached_left_notes[0][1].start//tick_per_beat,cached_left_notes,cached_right_notes)
                    for cached_note in cached_left_notes:
                        shift_up_count += 1
                        transpose_notes[cached_note[0]].pitch += 12
                        # all_left_notes[cached_note[0]].pitch += 12

        if (
            len(cached_left_notes) > 0
            and current_beat > cached_left_notes[0][1].start // tick_per_beat
        ):
            cached_left_notes = []
        if (
            len(cached_right_notes) > 0
            and current_beat > cached_right_notes[0][1].start // tick_per_beat
        ):
            cached_right_notes = []

        if closest_center == 0:
            # print('added to left')
            all_left_notes.append(note)
            all_right_notes.append(None)
            cached_left_notes.append([note_idx, note])
        else:
            # print(note, abs_distance_to_centers, closest_center, max_semitone_distance)
            # print('added to right')
            all_left_notes.append(None)
            all_right_notes.append(note)
            cached_right_notes.append([note_idx, note])
        if shiftbase:
            continue
        # ------------------end shift up----------------------------------

        if abs_distance_to_centers[closest_center] > max_semitone_distance:
            # transpose
            start_beat = (note.start) // tick_per_beat
            end_beat = (note.end - 1) // tick_per_beat
            assert start_beat <= end_beat

            if note.pitch > centers[closest_center]:
                # transpose downward
                if closest_center == 0:
                    # left hand
                    transpose_notes[note_idx].pitch -= 12 * (
                        (
                            (
                                abs_distance_to_centers[closest_center]
                                - max_semitone_distance
                            )
                            // 12
                        )
                        + 1
                    )
                    # print(note,range_per_beat[start_beat:end_beat+1,0],start_beat,end_beat,range_per_beat)
                    if transpose_notes[note_idx].pitch < max(
                        range_per_beat[start_beat : end_beat + 1, 0]
                    ):  # the tranposed note can never go beyond the skyline
                        transpose_notes[note_idx].pitch += 12
                        # -----------------------------remove sparse note----------
                        current_beat_notes = get_notes_of_beat(
                            transpose_notes, current_beat, tick_per_beat
                        )
                        current_beat_classified_notes, abs_dist = cluster_assignment(
                            current_beat_notes, centers_per_beat, tick_per_beat
                        )
                        if (
                            transpose_notes[note_idx].pitch
                            - min(
                                current_beat_classified_notes[0], key=lambda n: n.pitch
                            ).pitch
                            > sparse_threshold
                        ):
                            delete_sparse.add(note_idx)
                        # -------------end remove sparse note--------------------
                    else:
                        count += 1
                else:
                    # right hand
                    skip_count += 1
            else:
                # transpose upward
                if closest_center == 1:
                    # print('transposing',transpose_notes[note_idx], abs_distance_to_centers, closest_center, max_semitone_distance)
                    # right hand
                    transpose_notes[note_idx].pitch += 12 * (
                        (
                            (
                                abs_distance_to_centers[closest_center]
                                - max_semitone_distance
                            )
                            // 12
                        )
                        + 1
                    )
                    # print('transposed!!!',transpose_notes[note_idx], abs_distance_to_centers, closest_center, max_semitone_distance, min(range_per_beat[start_beat:end_beat+1,1]))
                    if transpose_notes[note_idx].pitch > min(
                        range_per_beat[start_beat : end_beat + 1, 1]
                    ):  # the tranposed note can never go beyond the skyline
                        transpose_notes[note_idx].pitch -= 12
                        # -----------------------------remove sparse note----------
                        current_beat_notes = get_notes_of_beat(
                            transpose_notes, current_beat, tick_per_beat
                        )
                        current_beat_classified_notes, abs_dist = cluster_assignment(
                            current_beat_notes, centers_per_beat, tick_per_beat
                        )
                        # print('----------------------------------------------------------------------------')
                        # print(tick_per_beat,current_beat,transpose_notes[note_idx],current_beat_notes,current_beat_classified_notes)
                        # print('~~~~')
                        # print(list(filter(lambda x:x.start>=transpose_notes[note_idx].start,transpose_notes))[:10])
                        if (
                            max(
                                current_beat_classified_notes[1], key=lambda n: n.pitch
                            ).pitch
                            - transpose_notes[note_idx].pitch
                            > sparse_threshold
                        ):
                            # print(max(current_beat_classified_notes[1],key=lambda n:n.pitch).pitch , transpose_notes[note_idx].pitch , sparse_threshold, transpose_notes[note_idx])
                            # print(current_beat, current_beat_classified_notes[1])
                            delete_sparse.add(note_idx)
                        # -------------end remove sparse note--------------------
                    else:
                        count += 1
                    # print('final@@@',transpose_notes[note_idx], abs_distance_to_centers, closest_center, max_semitone_distance)
                else:

                    # left hand
                    skip_count += 1

    # ------------remove sparse note -----------------------
    deleted = 0
    if deletesparse:
        transpose_notes = np.array(transpose_notes)
        # print(delete_sparse)
        transpose_notes = np.delete(transpose_notes, sorted(list(delete_sparse)))
        transpose_notes = list(transpose_notes)
        deleted = len(delete_sparse)
    # ---------------------end remove sparse note-----------

    # print('Octave Transpose:','#notes',len(transpose_notes),'transposed',count,'notes',', skipped',skip_count,'shifted_up',shift_up_count, 'deleted', deleted)
    # find the center again
    if shiftbase:
        centers_per_beat, range_per_beat = get_centers_per_beat(
            transpose_notes, tick_per_beat, center_threshold, mode=mode
        )
        all_left_notes = list(filter(lambda note: note is not None, all_left_notes))
        all_right_notes = list(filter(lambda note: note is not None, all_right_notes))
        # return transpose_notes,centers_per_beat,range_per_beat
        return all_left_notes, all_right_notes, centers_per_beat, range_per_beat

    # visualization : center line of left/right hand

    # plot(centers_per_beat)

    return transpose_notes, centers_per_beat, range_per_beat


def doubling_simplification(notes, centers_per_beat, tick_per_beat):
    notes = np.array(notes)
    count = 0
    track_tick = 0
    track_idx = 0
    collections = []
    delete = []
    threshold = 4  # max notes

    while track_idx < len(notes):
        if notes[track_idx].start == track_tick:
            # gathering notes start at the same tick
            collections.append(notes[track_idx])
            track_idx += 1
        else:
            # action
            classified, abs_distance_to_classified_centers = cluster_assignment(
                collections, centers_per_beat, tick_per_beat
            )
            keep = [[None for _ in range(12)] for __ in range(2)]
            # doubling simplification on different hand separately
            for group in range(2):
                note_count = 0
                delete_candidates = []
                for note in classified[group]:
                    if keep[group][note.pitch % 12] is None:
                        keep[group][note.pitch % 12] = note
                    elif keep[group][note.pitch % 12].pitch >= note.pitch:
                        delete_candidates.append(note)
                    else:
                        delete_candidates.append(keep[group][note.pitch % 12])
                        keep[group][note.pitch % 12] = note
                    note_count += 1
                if note_count > threshold:
                    # remove doubling
                    delete_count = note_count - threshold
                    if group == 0:
                        delete_candidates.sort(key=lambda x: x.pitch)
                    else:
                        delete_candidates.sort(key=lambda x: x.pitch, reverse=True)
                    delete_candidates = delete_candidates[0:delete_count]
                    for note in delete_candidates:
                        delete.append(note)
            # reset
            collections = [notes[track_idx]]
            track_tick = notes[track_idx].start
            track_idx += 1

    for note in delete:
        notes = np.delete(notes, np.where(notes == note)[0])
        count += 1

    # print('Doubling Simplification:','removed doubling',count)
    return notes


def merge_discrete_note(notes, tick_per_beat, merge_threshold):
    notes.sort(key=lambda note: note.start)
    org_len = len(notes)
    # profile
    all_pitch = {}
    for note in notes:
        pitch = note.pitch
        if pitch in all_pitch:
            all_pitch[pitch].append(note)
        else:
            all_pitch[pitch] = [note]
    # merge
    for k, v in all_pitch.items():
        v.sort(key=lambda note: note.start)
        trace_idx = 0
        while trace_idx < len(v) - 1:
            if (
                0
                <= v[trace_idx + 1].start - v[trace_idx].end
                <= merge_threshold * tick_per_beat
            ):
                v[trace_idx].end = v[trace_idx + 1].end
                v.pop(trace_idx + 1)
            else:
                trace_idx += 1
    # output
    notes = []
    for k, v in all_pitch.items():
        for note in v:
            notes.append(note)
    notes.sort(key=lambda note: note.start)
    filtered_len = len(notes)
    # print('Merge discrete:','merged ',org_len-filtered_len,' discrete notes')
    return notes


def drop_discrete_note(notes, tick_per_beat, discrete_note_threshold):
    org_len = len(notes)

    new_notes = []
    for note in notes:
        if note.end - note.start > discrete_note_threshold * tick_per_beat:
            new_notes.append(note)
    notes = new_notes
    # print('Drop discrete:','dropped ',org_len-len(notes),' discrete notes')
    return notes


def postprocess(
    inpath,
    outpath,
    merge_threshold=0.00005,
    discrete_note_threshold=0.0005,
    merge=False,
):
    path = inpath
    notes, tick_per_beat = read_midi(path)
    org_notes = len(notes)
    # print(f'before post-processing {org_notes} notes')
    notes = trim_long_notes(notes, tick_per_beat)
    notes, centers_per_beat, range_per_beat = octave_transpose(
        notes, tick_per_beat, 3, 6, mode="normal"
    )
    if merge:
        notes = merge_discrete_note(notes, tick_per_beat, merge_threshold)
    notes = drop_discrete_note(notes, tick_per_beat, discrete_note_threshold)
    notes = doubling_simplification(notes, centers_per_beat, tick_per_beat)
    cur_notes = len(notes)
    # print(f'after post-processing, total notes: {cur_notes}, removed {org_notes-cur_notes} notes in total')
    # visualize(notes,tick_per_beat,centers_per_beat=centers_per_beat,range_per_beat=None,length=5000)
    write_midi(notes, centers_per_beat, outpath, tick_per_beat)

    minimum_runs = 3
    run = 0
    while True:
        notes, tick_per_beat = read_midi(outpath)
        org_notes = len(notes)
        # print(f'before post-processing {org_notes} notes')
        notes = trim_long_notes(notes, tick_per_beat)
        notes, centers_per_beat, range_per_beat = octave_transpose(
            notes, tick_per_beat, 1, 10, range_per_beat, mode="centroid"
        )
        if merge:
            notes = merge_discrete_note(notes, tick_per_beat, merge_threshold)
        notes = drop_discrete_note(notes, tick_per_beat, discrete_note_threshold)
        notes = doubling_simplification(notes, centers_per_beat, tick_per_beat)
        cur_notes = len(notes)
        # print(f'after post-processing, total notes: {cur_notes}, removed {org_notes-cur_notes} notes in total')
        # visualize(notes,tick_per_beat,centers_per_beat=centers_per_beat,range_per_beat=None,length=5000)
        write_midi(notes, centers_per_beat, outpath, tick_per_beat)
        run += 1
        # termination
        if org_notes - cur_notes == 0 and run >= minimum_runs:
            break

    notes, tick_per_beat = read_midi(outpath)
    org_notes = len(notes)
    # print(f'before post-processing {org_notes} notes')
    # delete sparse
    notes, centers_per_beat, range_per_beat = octave_transpose(
        notes, tick_per_beat, 0, 1, range_per_beat, mode="minmax", deletesparse=True
    )
    # shiftbase
    notes_lh, notes_rh, centers_per_beat, range_per_beat = octave_transpose(
        notes, tick_per_beat, 0, 1, range_per_beat, mode="minmax", shiftbase=True
    )
    cur_notes = len(notes)
    # print(f'after post-processing, total notes: {cur_notes}, removed {org_notes-cur_notes} notes in total')
    # visualize(notes,tick_per_beat,centers_per_beat=centers_per_beat,range_per_beat=None,length=5000)
    write_midi(notes_lh, centers_per_beat, outpath, tick_per_beat, notes_rh)
    logger.info("output to:", outpath, "#notes:", cur_notes)
