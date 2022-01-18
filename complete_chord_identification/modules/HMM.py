import numpy as np
import random
import itertools
import math
import tqdm
from tqdm import trange
import time
import pickle
import noteToChordWeighted  as noteToChord
key_mapping={
    'C':0,
    'D':2,
    'E':4,
    'F':5,
    'G':7,
    'A':9,
    'B':11
}
def key2num(key):  
  key=key.upper()
  num=key_mapping[key[0]]
  modifier=len(key)
  if modifier==1:
    return num
  elif key[1]=='#':
    return (num+(modifier-1))%12
  elif key[1]=='B' or key[1]=='-':
    return (num-(modifier-1))%12
  elif key[1]=='X':
    return (num+(modifier-1)*2)%12

# key_list to number_list
def keys2num(keys):
    return [key2num(key) for key in keys]

# function to distingish whether notes in given timestamp and chord are within or outside the chord (0=outside,1=within)
import json
with open('./modules/json_files/keychorddict.json') as json_file:
    chord_notes = json.load(json_file)
  
    def note_2_class(chord,notes_at_t,chord_notes=chord_notes):
        notes_at_t=keys2num(notes_at_t)
        note_in_chord=chord_notes[chord]['idx']
        return [int(note in note_in_chord) for note in notes_at_t]

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

    return [key2num(key) for key in keys]



class HMM:
    def __init__(self,no_of_state,no_of_value,states,values):
        #randomize all matrix, each row should sum to 1
        #self.emssion_matrix=np.array([ran/ran.sum() for ran in np.array([np.random.rand(no_of_value) for i in range(no_of_state)])]) #b[i][O]  --> probability to emit values[O] from states[i]
        

        self.zero=0.00000000000001
        self.emssion_matrix=np.random.rand(no_of_value)
        self.emssion_matrix=self.emssion_matrix/self.emssion_matrix.sum()
        if self.emssion_matrix[0]>self.emssion_matrix[1]:
            temp=self.emssion_matrix[0]
            self.emssion_matrix[0]=self.emssion_matrix[1]
            self.emssion_matrix[1]=temp
        self.emssion_matrix=np.array([self.emssion_matrix,]*len(states)) 
        self.emssion_matrix=np.array([[0.2,0.8],]*len(states)) 
        # assume same probability to emit "note within chord"  for all chords
        
        self.initial_matrix= np.random.rand(no_of_state) #π[i] --> probability to start at states[i]
        self.initial_matrix/= self.initial_matrix.sum()
        
        self.transition_matrix=np.array([ran/ran.sum() for ran in np.array([np.random.rand(no_of_state) for i in range(no_of_state)])])  #a[i][j] --> probability from states[i] transit to states[j]
        self.no_of_state=no_of_state
        self.states=states
        self.values=values
        self.observered=None
        self.key=None
        
        self.probit_at_i_table=None
        self.probit_transit_i_j_table=None
        self.forward_table=None
        self.backward_table=None
        self.chord_probit=None
        self.note_probit=None
        
    def debug(self):
        print('initial_matrix\n',self.initial_matrix)
        print('transition_matrix\n',self.transition_matrix)
        print('emission_matrix\n',self.emssion_matrix)
        print("note prob\n",self.note_probit)
    
    def likelihood(self,state,ob_t,key):
        
        #profile
        majorP=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minorP=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        majorP/=majorP.sum()
        minorP/=minorP.sum()
        #key mask
        if key[-5:].lower()!=self.states[state][:5].lower():
            return np.log(self.zero)
        
        chord=key+self.states[state][5:]

        #prediction from note2chord
        a=noteToChord.NoteToChord(ob_t,key,20,0)
        #print(len(a),a)
  
        #verify output number
        if key[-5:]=='Major':
            if(not(len(a)==12)):
                print(a,ob_t,key)
            assert(len(a)==12)
        else:
            if(not(len(a)==15)):
                print(a,ob_t,key)
            assert(len(a)==15)
       
#         seventh=['MinorII',
#                 'MinorV+',
#                 'MinorDimVII',
#                 'MajorII',
#                 'MajorV',
#                 'MajorVII']#,'MajorDimVII'
        chords=[]
        score=[]
        
        for idx,e in enumerate(a):
            
#             if sum([s in e['Chord'] for s in seventh]):
#                 if e['hasSeventh']:
#                     chords.append(e['Chord']+'7')
#                     score.append(e['Score'])
#                 else:
#                     chords.append(e['Chord']+'7')
#                     score.append(0)
            chords.append(e['Chord'])
            score.append(e['Score'])
        score=np.array(score)
        
        #normalize
        score/=sum(score)
        score=[max(score)-score[e] for e in range(len(score))]
        score=[(np.exp(-100*i)) for i in score] #fixed template probit
        score/=sum(score)
        score=[e if e>0 else self.zero for e in score]


        

        #get the idx of interested chord(state)
        #chords=[i['Chord']+'7' if i['hasSeventh'] else i['Chord'] for i in a]
        #print(chords)
        if chord[:7]=='dbMinor':
            chord=chord.replace('dbMinor','c#Minor')
        elif chord[:7]=='abMinor':
            chord=chord.replace('abMinor','g#Minor')
        elif chord[:7]=='F#Major':
            chord=chord.replace('F#Major','GbMajor')
        idx=chords.index(chord)
        
        #probit to emit chordNote/non-chordNote
        observation=ob_t.keys()
        obs_weight=[ob_t[e] for e in observation]
      #  obs_2=note_2_class(chord,observation) #2 class
      #  prob=0
        note_prob=0
        obs_no=keys2num(observation)
        for i,x in enumerate(obs_no):
            note_prob+=self.note_probit[x]*obs_weight[i]
     #   for i,x in enumerate(obs_2):
      #      prob+=(self.emssion_matrix[state][x])*obs_weight[i]
        
        prob_note_given_key=0

        key_no=keys2num([key[:-5]])[0]
        
        if key[-5:]=='Major':
            for i,x in enumerate(obs_no):
                prob_note_given_key+=majorP[x-key_no]*obs_weight[i]
        else:
            for i,x in enumerate(obs_no):
                prob_note_given_key+=minorP[x-key_no]*obs_weight[i]

        assert(score[idx]>0)
        #assert(prob>0)
            #   N2C P(chord|note)*  P(note)        / P(chord)
        #return np.log(score[idx])+np.log(note_prob)-np.log(self.chord_probit[state]) # +np.log(prob)
        '''
        print(ob_t.keys())
        print(obs_weight)
        print(self.states[state])
        print(key_no)
        print(chords)
        print(prob_note_given_key)
        print(score)
        print(idx)
        '''
        
        return np.log(prob_note_given_key)+np.log(score[idx])-np.log(self.chord_probit[state])
        
    def forward(self,t,j,ob=None,mode=False):
        if ob is None:
            ob=self.observered
        if np.isnan(self.forward_table[t][j]).any():
            if t==0:
                if mode==True:
                    self.forward_table[t][j]=[(np.log(self.initial_matrix[j]))+self.likelihood(j,ob[t],self.key[t]),0]
                    return self.forward_table[t][j][0],self.forward_table[t][j][1]
            else:          
                if mode==True:                
                    if self.key[t]==self.key[t-1]:
                        result=np.array([self.forward(t-1,i,ob,mode)[0]+np.log(self.transition_matrix[i][j]) for i in range(self.no_of_state)])+self.likelihood(j,ob[t],self.key[t])
                    else:
                        result=np.array([self.forward(t-1,i,ob,mode)[0] for i in range(self.no_of_state)])+self.likelihood(j,ob[t],self.key[t])
                    self.forward_table[t][j]=[np.max(result),np.argmax(result)]
                    return self.forward_table[t][j][0],self.forward_table[t][j][1]
        else:
            return self.forward_table[t][j]
    
    def backward(self,t,i,ob=None):
        if ob is None:
            ob=self.observered
        if np.isnan(self.backward_table[t][i]):
            if t==len(ob)-1:
                self.backward_table[t][i]=1
                return self.backward_table[t][i]
            else:
                self.backward_table[t][i]=sum([self.transition_matrix[i][j]*self.likelihood(j,ob[t+1])*self.backward(t+1,j,ob) for j in range(self.no_of_state)])
                return self.backward_table[t][i]
        else:
            return self.backward_table[t][i]
            
           
    def probit_at_i(self,t,i,ob=None):#Gamma γt(i) = P(qt = i|O,λ)      
        if ob is None:
            ob=self.observered
        if np.isnan(self.probit_at_i_table[t][i]):
            numerator=self.forward(t,i,ob)*self.backward(t,i,ob)#sum probability of all path passing through state[i] at time t
            denominator=sum([self.forward(t,j,ob)*self.backward(t,j,ob) for j in range(self.no_of_state)]) #prob of passing through  ALL_state at time t
            self.probit_at_i_table[t][i]=numerator/denominator
            return self.probit_at_i_table[t][i]
        else:
            return self.probit_at_i_table[t][i]
    
    def probit_transit_i_j(self,t,i,j,ob=None):#epsilon ξt(i, j) = P(qt = i,qt+1 = j|O,λ)
        if ob is None:
            ob=self.observered
        if np.isnan(self.probit_transit_i_j_table[t][i][j]):
            numerator=self.forward(t,i,ob)*self.transition_matrix[i][j]*self.likelihood(j,ob[t+1])*self.backward(t+1,j,ob)#sum probability of all path transit from state[i] to state[j] at time t
            denominator=sum([sum([self.forward(t,m,ob)*self.transition_matrix[m][n]*self.likelihood(n,ob[t+1])*self.backward(t+1,n,ob) for n in range(self.no_of_state)]) for m in range(self.no_of_state)]) #prob of ALL transition combination at time t
            self.probit_transit_i_j_table[t][i][j]=(numerator/denominator)
            return self.probit_transit_i_j_table[t][i][j]
        else:
            return self.probit_transit_i_j_table[t][i][j]
    
    #modify from https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm/9730083 , author RBF06(https://stackoverflow.com/users/3311728/rbf06)
    def predict(self,ob,test_key):  
        self.forward_table=np.empty((len(ob),self.no_of_state,2))
        self.forward_table[:]= np.NaN
        self.key=test_key
        self.chord_probit=self.transition_matrix.sum(axis=0)
        self.chord_probit[:18]/=self.chord_probit[:18].sum()
        self.chord_probit[18:]/=self.chord_probit[18:].sum()
  
 

        T1=np.empty((self.no_of_state,len(ob)),'d')
        T2=np.empty((self.no_of_state,len(ob)),'B')
        for idx in range(self.no_of_state):
            T1[idx,0]=self.forward(0,idx,ob,True)[0]
        T2[:,0]=0
        
        for i in range(1,len(ob)):
            for idx in range(self.no_of_state):
                T1[idx,i],T2[idx,i]=self.forward(i,idx,ob,True)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        x = np.empty(len(ob), 'B')
        x[-1] = np.argmax(T1[:, len(ob) - 1])
        
        #backtracking
        for i in reversed(range(1, len(ob))):
            x[i - 1] = T2[x[i], i]
        return x
    
    def get_detail(self,label):
        key='Major' if 'M' in label else 'Minor'
        chord=label.split("_")[1]
        key_note=label.split('M')[0] if key=='Major' else label.split('m')[0].lower()
        if key=='Major':
            if key_note=='F#':
                key_note='Gb'
            elif key_note=='A#':
                key_note='Bb'
            elif key_note=='C#':
                key_note='Db'
            elif key_note=='G#':
                key_note='Ab'
            elif key_note=='D#':
                key_note='Eb'
        else:
            if key_note=='gb':
                key_note='f#'
            elif key_note=='db':
                key_note='c#'
            elif key_note=='ab':
                key_note='g#'
            elif key_note=='d#':
                key_note='eb'
                
        if 'German' in chord:
            chord=key+'GerVI'
        elif 'Dim7' in chord:
            chord=key+'DimVII7'
        elif 'FrenchVI' in chord:
            chord=key+'FreVI'
        else:
            chord=key+chord
            #MajorDimVII7
        if 'MajorDimVII' in chord:
            chord=chord.replace('MajorDimVII','MajorVII')
        if '7' in chord:
            chord=chord.replace('7','')
  
        return chord,key_note
    
    def train_supervisied(self,obs,labels): # by MLE
        
        initial_matrix= np.zeros(self.no_of_state)
        emssion_matrix=np.zeros((2)) 
        transition_matrix=np.array([np.zeros(self.no_of_state) for i in range(self.no_of_state)])
        note_list=np.zeros(12)

        for idx_1,label in enumerate(labels):#loop each score
            for idx_2,lab in enumerate(label):

                chord,key_name=self.get_detail(label[idx_2])
                if idx_2==0:
                    initial_matrix[self.states.index(chord)]+=1
                else:
                    Pre_chord,Pre_key_name=self.get_detail(label[idx_2-1])
                    if '+' in Pre_chord and 'Major' in Pre_chord:
                        Pre_chord=Pre_chord.replace('+','')
                    if '+' in chord and 'Major' in chord:
                        chord=chord.replace('+','')
                        
                    
                    transition_matrix[self.states.index(Pre_chord)][self.states.index(chord)]+=1
                    ob_t_no=keys2num(obs[idx_1][idx_2])
                 #   ob_t=note_2_class(key_name+chord,obs[idx_1][idx_2]) #2 class
               #     for x in ob_t:
              #          emssion_matrix[x]+=1
                    for x in ob_t_no:
                        note_list[x]+=1
                        
        #save back to model
        self.initial_matrix=np.array([item/initial_matrix.sum() if item >0 else self.zero for item in initial_matrix])
        self.transition_matrix=np.array([row/row.sum() if row.sum()>0 else row+self.zero for row in transition_matrix])
        
        #import transition calculated from ABC dataset
        #with open('C:/Users/tokah/Documents/fyp-chord-identification/data/beet_Transition', 'rb') as handle:
            #b_transition = pickle.load(handle)
        #self.transition_matrix=b_transition*0.1+self.transition_matrix*0.9
        
        self.transition_matrix=np.array([row/row.sum() if row.sum()>0 else row+self.zero for row in self.transition_matrix])
        
        
        for row in range(self.transition_matrix.shape[0]):
            for col in range(self.transition_matrix.shape[1]):
                if self.transition_matrix[row][col]==0:
                    self.transition_matrix[row][col]=self.zero
     #   self.emssion_matrix=np.array([emssion_matrix/emssion_matrix.sum()] *self.no_of_state) 
        self.note_probit=note_list/sum(note_list)

      
    def train(self,obs,key_name,epochs=2):
        #O:observed values
        #λ:model parameters

        
        for epoch in range(epochs):
            for ob_idx,ob in enumerate(obs):

                print('running',ob_idx,key_name[ob_idx].lower()[:-1])
                
                self.probit_at_i_table=np.empty((len(ob),self.no_of_state))
                self.probit_transit_i_j_table=np.empty((len(ob),self.no_of_state,self.no_of_state))
                self.forward_table=np.empty((len(ob),self.no_of_state))
                self.backward_table=np.empty((len(ob),self.no_of_state))
                self.probit_at_i_table[:]= np.NaN
                self.probit_transit_i_j_table[:]=np.NaN
                self.forward_table[:]= np.NaN
                self.backward_table[:]=np.NaN

                #initalize DP table
                for t, i in tqdm.tqdm(itertools.product(range(len(ob)),range(self.no_of_state))):
                    self.forward(t,i,ob)
                    self.backward(t,i,ob)
                    self.probit_at_i(t,i,ob)
                    for j in range(self.no_of_state):
                        if t!=len(ob)-1:
                            self.probit_transit_i_j(t,i,j,ob)
                
                
                
                #initial matrix
                #for i in range(self.no_of_state):
                #    self.initial_matrix[i]=self.probit_at_i(0,i,ob)
                #    if self.initial_matrix[i]==0:
                #        self.initial_matrix[i]=0.00000001
                
                #transition matrix
                for i, j in itertools.product(range(self.no_of_state),range(self.no_of_state)):
                    self.transition_matrix[i][j]=sum([self.probit_transit_i_j(t,i,j,ob) for t in range(len(ob)-1)])/sum([self.probit_at_i(t,i,ob) for t in range(len(ob)-1)])
                    if self.transition_matrix[i][j]==0 :
                        self.transition_matrix[i][j]=self.zero
                #emission matrix
                #for j, k in itertools.product(range(self.no_of_state),range(len(self.values))):   
                #    total=0
                #    
                #    #Modification: convert notes to 2 classes (outside or inside given chord)
                #    chord=self.states[j]  #numeric to chord name
                #    ob_2_class=[note_2_class(chord,ob_t) for ob_t in ob] #2 class

                #    for t in range(len(ob)):
                #        if k in ob_2_class[t]:
                #            #Modification: multiple by how many times do k appear at timestamp t
                #            total+=self.probit_at_i(t,j,ob)*ob_2_class[t].count(k)  
                            
                #    #Modification: multiple by len(ob[t]), which is the total length of notes at timestamp t                    
                #    self.emssion_matrix[:,k]=total/sum([self.probit_at_i(t,j,ob)*len(ob[t]) for t in range(len(ob))])  
                    
                #    #smoothing
                    
                #    if self.emssion_matrix[j][k]==0 :
                #        self.emssion_matrix[:,k]=0.00000001
                     