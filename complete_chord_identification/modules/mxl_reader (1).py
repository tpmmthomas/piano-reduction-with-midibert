import streamlit as st
import subprocess
import base64
import os
import shutil
import copy
import csv
import math
import music21 as m21
#from music21.converter.subConverters import ConverterMusicXML
#from DivideByBeat import *
from MusicXML import *

chord_to_num = [
    {
        'I': 0,
        'II': 1,
        'II7': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
        'VII7': 8,
        'V7': 9,
        '♭VI': 10,
        '♭II': 11,
        'dim VII': 12,
        'dim VII7': 13,
        'N6': 14,
        'It6': 15,
        'Fr6': 16,
        'Ger6': 17
    },
    {
        'I': 0,
        'II': 1,
        'II7': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
        'VII7': 8,
        '♭II': 9,
        'dim VII': 10,
        'dim VII7': 11,
        'N6': 12,
        'It6': 13,
        'Fr6': 14,
        'Ger6': 15,
        'I+': 16,
        'IV+': 17,
        'V+': 18,
        'V+7': 19
    },
]

chord_not_exist = []

# Write the csv file with certain format
def write_csv_file(file_name, beats_info):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = []
        for i in range(12):
            header.append(i)
        header += ['Key', 'M/m', 'Chord']
        writer.writerow(header)
        for beat_info in beats_info:
            writer.writerow(beat_info)

# extract beat info
def extract_mxl_info(file_name):

    xml_file = m21.converter.parse(file_name)

    measure_duration = xml_file.parts[0].measure(1).duration.quarterLength
    beat_count = xml_file.parts[0].flat.timeSignature.beatCount
    # important for doing segmentation
    beat_duration = measure_duration / beat_count
    measure_num = get_measure_count(xml_file)

    current_key = '?'
    current_chord = '?'

    beats_duration_info = []
    beats_duration_normalized_info = []
    beats_frequency_info = []
    beats_frequency_normalized_info = []
    beats_duration_times_frequency_info = []
    beats_duration_times_frequency_normalized_info = []

    for measure in range(0, measure_num):
        
        # notes_info saves: [{note degree[0 - 11]}, {start time}, {end time}]
        # key_chord_change_info saves: [{key}, {chord}, {start time}]
        note_duration_info = []
        key_chord_change_info = []

        measure_notes = extract_notes_in_measures(xml_file, measure, measure)
        
        max_note_end_time = 0
        for note in measure_notes:
            # not to consider the rest
            if note.isRest == False:
                note_end_time = note.offset + float(note.quarterLength)
                if note_end_time >= max_note_end_time:
                    max_note_end_time = note_end_time
                if note.lyric is not None:
                    if '(' in note.lyric:
                        current_key = note.lyric.split('(')[0]
                        current_chord = note.lyric.split('(')[-1][:-1]
                    else:
                        current_chord = note.lyric
                    key_chord_change_info.append([current_key, current_chord, note.offset])
                if len(key_chord_change_info) == 0:
                    key_chord_change_info.append([current_key, current_chord, 0.0])
                if current_key == '?':
                    note_duration_info.append([-1, note.offset, note_end_time])
                else:
                    m21_key = current_key.replace('♭', '-').replace('M', '').replace('m', '')
                    note_duration_info.append([(note.pitch.midi - m21.pitch.Pitch(m21_key).midi) % 12, note.offset, note_end_time])

        beat_info_temp = []
        beat_segment_num = math.ceil(max_note_end_time / beat_duration)
        # initialize beat info
        for i in range(0, beat_segment_num):
            temp_list = [0] * 15
            beat_info_temp.append(temp_list)

        # update key, tonailty, chord info
        for info in key_chord_change_info:
            index = int(info[2] / beat_duration)
            m21_key = info[0].replace('♭', '-').replace('M', '').replace('m', '')
            if m21_key == '?':
                key_num = -1
                tonality = -1
            else:    
                key_num = m21.pitch.Pitch(m21_key).midi % 12
                if info[0][-1] == 'M':
                    tonality = 0
                else:
                    tonality = 1
            beat_info_temp[index][12] = key_num
            beat_info_temp[index][13] = tonality
            # To Modify
            if tonality != -1:
                if info[1] in chord_to_num[tonality].keys():
                    beat_info_temp[index][14] = chord_to_num[tonality][info[1]]
                else:
                    if info[1] not in chord_not_exist:
                        #st.write(info[1])
                        chord_not_exist.append(info[1])
                    beat_info_temp[index][14] = -1
            else:
                beat_info_temp[index][14] = -1
        temp_index = 0
        for i in range(0, beat_segment_num):
            if beat_info_temp[i][14] != 0:
                temp_index = i
            else:
                beat_info_temp[i][12] = beat_info_temp[temp_index][12]
                beat_info_temp[i][13] = beat_info_temp[temp_index][13]
                beat_info_temp[i][14] = beat_info_temp[temp_index][14]
        
        beat_duration_info = copy.deepcopy(beat_info_temp)
        beat_duration_normalized_info = copy.deepcopy(beat_info_temp)
        beat_frequency_info = copy.deepcopy(beat_info_temp)
        beat_frequency_normalized_info = copy.deepcopy(beat_info_temp)
        beat_duration_times_frequency_info = copy.deepcopy(beat_info_temp)
        beat_duration_times_frequency_normalized_info = copy.deepcopy(beat_info_temp)

        # update beat_duration and beat_frequency info
        for info in note_duration_info:
            if info[0] >= 0:
                start_beat = int(info[1] / beat_duration)
                end_beat = int((info[2] - 0.00001) / beat_duration)
                if start_beat == end_beat:
                    beat_duration_info[start_beat][info[0]] += info[2] - info[1]
                    beat_frequency_info[start_beat][info[0]] += 1
                else:
                    beat_duration_info[start_beat][info[0]] += (start_beat + 1) * beat_duration - info[1]
                    beat_frequency_info[start_beat][info[0]] += 1
                    beat_duration_info[end_beat][info[0]] += info[2] - end_beat * beat_duration
                    beat_frequency_info[end_beat][info[0]] += 1
                    if end_beat - start_beat > 1:
                        for i in range(start_beat + 1, end_beat):
                            beat_duration_info[i][info[0]] += beat_duration
                            beat_frequency_info[i][info[0]] += 1

        # update duration times frequency info
        for i in range(0, beat_segment_num):
            for j in range(0, 12):
                beat_duration_times_frequency_info[i][j] = beat_duration_info[i][j] * beat_frequency_info[i][j]

        # normalize duration and frequency
        for i in range(0, beat_segment_num):
            all_notes_duration = 0
            all_notes_frequency = 0
            all_notes_duration_times_frequency = 0
            for j in range(0, 12):
                all_notes_duration += beat_duration_info[i][j]
                all_notes_frequency += beat_frequency_info[i][j]
                all_notes_duration_times_frequency += beat_duration_times_frequency_info[i][j]
            for j in range(0, 12):
                if all_notes_duration > 0.00001:
                    beat_duration_normalized_info[i][j] = beat_duration_info[i][j] / all_notes_duration
                else:
                    beat_duration_normalized_info[i][j] = 0
                if all_notes_frequency > 0:
                    beat_frequency_normalized_info[i][j] = beat_frequency_info[i][j] / all_notes_frequency
                else:
                    beat_frequency_normalized_info[i][j] = 0
                if all_notes_duration_times_frequency > 0.00001:
                    beat_duration_times_frequency_normalized_info[i][j] = beat_duration_times_frequency_info[i][j] / all_notes_duration_times_frequency
                else:
                    beat_duration_times_frequency_normalized_info[i][j] = 0

        #print(measure)
        for i in range(0, beat_segment_num):
            beats_duration_info.append(beat_duration_info[i])
            beats_duration_normalized_info.append(beat_duration_normalized_info[i])
            beats_frequency_info.append(beat_frequency_info[i])
            beats_frequency_normalized_info.append(beat_frequency_normalized_info[i])
            beats_duration_times_frequency_info.append(beat_duration_times_frequency_info[i])
            beats_duration_times_frequency_normalized_info.append(beat_duration_times_frequency_normalized_info[i])
        #print("")

    write_csv_file(file_name.split('.mxl')[0] + '_duration.csv', beats_duration_info)
    write_csv_file(file_name.split('.mxl')[0] + '_duration_normalized.csv', beats_duration_normalized_info)
    write_csv_file(file_name.split('.mxl')[0] + '_frequency.csv', beats_frequency_info)
    write_csv_file(file_name.split('.mxl')[0] + '_frequency_normalized.csv', beats_frequency_normalized_info)
    write_csv_file(file_name.split('.mxl')[0] + '_duration_times_frequency.csv', beats_duration_times_frequency_info)
    write_csv_file(file_name.split('.mxl')[0] + '_duration_times_frequency_normalized.csv', beats_duration_times_frequency_normalized_info)

if __name__ == '__main__':

    # File upload setting
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Title
    st.title("MXL File Reader by KY2103")
    st.write("")

    # Upload mxl file
    input_file = st.file_uploader("Upload the .mxl file", type=['mxl', 'zip'], key='input')

    if input_file is not None:
        if st.button("Start"):
            st.write("Start extracting...")

            # Create local temp directory
            source_folder_path = input_file.name[:-4] + '/'
            os.mkdir(source_folder_path)

            # Save the input file into local directory
            with open(source_folder_path + input_file.name, "wb") as f:
                f.write(input_file.getbuffer())
                f.close()

            # Extract file names
            source_file_name_list = []
            if input_file.name.split('.')[-1] == 'zip':
                subprocess.call("unzip ./{} -d ./{}".format(source_folder_path + input_file.name, source_folder_path), shell=True)
                for file_name in os.listdir(source_folder_path):
                    if file_name.split('.')[-1] == 'mxl':
                        source_file_name_list.append(file_name)
                os.remove(source_folder_path + input_file.name)
            else:
                source_file_name_list.append(input_file.name)

            # Extract mxl info
            source_file_num = len(source_file_name_list)

            for i in range(0, source_file_num):
                st.write("({}/{}) Extracting {}...".format(str(i + 1), str(source_file_num), source_file_name_list[i]))
                extract_mxl_info(source_folder_path + source_file_name_list[i])
                os.remove(source_folder_path + source_file_name_list[i])
                st.write("Finished extracting {}!".format(source_file_name_list[i]))
            
            
            st.write("Zipping the output files...")
            # Zip the extracted csv file
            subprocess.call("zip -r ./{}_extracted.zip ./{}".format(input_file.name[:-4], source_folder_path), shell=True)
            
            # Let user to download
            with open("./{}_extracted.zip".format(input_file.name[:-4]), "rb") as f:
                bytes = f.read()
                b64 = base64.b64encode(bytes).decode()
                href = f'<a href="data:file/obj;base64,{b64}" download="{input_file.name[:-4]}_extracted.zip">{input_file.name[:-4]}_extracted.zip</a>'
                st.write("You can download the result zip file:")
                st.markdown(href, unsafe_allow_html=True)
            
            # Remove the local temp path
            shutil.rmtree(source_folder_path, ignore_errors=True)
            st.write("Finished zipping {}!".format(source_file_name_list[i]))
            st.success('Finish!')

            # Show the chords that are not indicated
            st.write('The following chords are not indicated from our chord table:')
            st.write(chord_not_exist)
