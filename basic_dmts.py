import nengo
import nengo.spa as spa
import numpy as np


T_isi = 0.5      # time between trials
T_present = 0.5  # time to present target stimulus
T_delay = 1.0    # delay time
T_respond = 1.0  # time allowed for response

similarity = 0   # how similar the target and foil are (0 to 1)
acc_scale = 0.2  # scaling factor on the accumulator

D = 32           # number of dimensions in the representation
    
    
# Code for generating the experimental stimuli
class Stimulus(object):
    def __init__(self, dimensions, similarity, seed=None):
        self.items = []
        self.rng = np.random.RandomState(seed=seed)
        self.vocab = spa.Vocabulary(dimensions)
        self.dimensions = dimensions
        self.similarity = similarity
        self.zero = np.zeros(dimensions)
    def make_pair(self):
        a = spa.SemanticPointer(D, rng=self.rng)
        b = spa.SemanticPointer(D, rng=self.rng)
        b = (similarity)*a + (1-similarity)*b
        return a, b
    def get_trial_info(self, t):
        T = T_isi + T_present + T_delay + T_respond
        index = int(t / T)
        t = t % T
        if t < T_isi:
            phase = 'isi'
        elif t < T_isi + T_present:
            phase = 'present'
        elif t < T_isi + T_present + T_delay:
            phase = 'delay'
        else:
            phase = 'respond'
        while len(self.items) <= index:
            a, b = self.make_pair()
            shown = self.rng.choice(['a', 'b'])
            self.vocab.add('A%d' % len(self.items), a)
            self.vocab.add('B%d' % len(self.items), b)
            self.items.append((a, b, shown))
        return self.items[index], phase
        
    def target(self, t):
        (a, b, shown), phase = self.get_trial_info(t)
        if phase == 'present':
            if shown == 'a':
                return a.v
            else:
                return b.v
        else:
            return self.zero
            
    def choice_a(self, t):
        (a, b, shown), phase = self.get_trial_info(t)
        if phase == 'respond':
            return a.v
        else:
            return self.zero
            
    def choice_b(self, t):
        (a, b, shown), phase = self.get_trial_info(t)
        if phase == 'respond':
            return b.v
        else:
            return self.zero
    
    def reset(self, t):
        (a, b, shown), phase = self.get_trial_info(t)
        if phase == 'respond':
            return 0
        else:
            return 1
        
stim = Stimulus(D, similarity)

# define the model
model = spa.SPA()
with model:
    
    # neurons to represent all the stimuli
    
    stim_target = nengo.Node(stim.target)
    model.target = spa.State(D, vocab=stim.vocab)
    nengo.Connection(stim_target, model.target.input, synapse=None)
    
    stim_choice_a = nengo.Node(stim.choice_a)
    model.choice_a = spa.State(D, vocab=stim.vocab)
    nengo.Connection(stim_choice_a, model.choice_a.input, synapse=None)

    stim_choice_b = nengo.Node(stim.choice_b)
    model.choice_b = spa.State(D, vocab=stim.vocab)
    nengo.Connection(stim_choice_b, model.choice_b.input, synapse=None)
    
    # memory 
    model.memory = spa.State(D, vocab=stim.vocab, feedback=1)
    nengo.Connection(model.target.output, model.memory.input)
    
    # comparison systems
    model.compare_a = spa.Compare(D)
    nengo.Connection(model.memory.output, model.compare_a.inputA)
    nengo.Connection(model.choice_a.output, model.compare_a.inputB)

    model.compare_b = spa.Compare(D)
    nengo.Connection(model.memory.output, model.compare_b.inputA)
    nengo.Connection(model.choice_b.output, model.compare_b.inputB)
    
    # accumulator to produce output   
    accumulator = nengo.Ensemble(n_neurons=200, dimensions=1)
    nengo.Connection(accumulator, accumulator, synapse=0.1)
    nengo.Connection(model.compare_a.output, accumulator,
                     transform=-acc_scale)
    nengo.Connection(model.compare_b.output, accumulator,
                     transform=acc_scale)
                     
    # reset the accumulator between trials
    stim_reset = nengo.Node(stim.reset)
    nengo.Connection(stim_reset, accumulator.neurons,
                     transform = -3*np.ones((accumulator.n_neurons, 1)))
