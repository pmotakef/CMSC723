from __future__ import print_function

from collections import namedtuple

Entry = namedtuple("Entry", ["index", "word", "lemma", "coarsePOS", "finePOS", "x1", "head", "deptype", "x2", "x3"])
Arc = namedtuple("Arc", ["head_idx", "tail_idx", "head", "tail"])


# Define feature functions
#
# For features that involve stack, there is a "try-except" block for cases
# when the stack is empty (the last step in a successful parse, when you just
# made a right-arc from the ROOT and shifted it to the buffer.) The buffer
# being empty indicates the end of the parse, so there are no additional checks
# in those functions

def ST_EDGE_W(stack, buf):
    try:
        return (("ST_EDGE_W", stack[-1].word))
    except:
        return (("ST_EDGE_W", ""))


def BUF_EDGE_W(stack, buf):
    return (("BUF_EDGE_W", buf[0].word))


def ST_EDGE_POS(stack, buf):
    try:
        return (("ST_EDGE_POS", stack[-1].coarsePOS))
    except:
        return (("ST_EDGE_POS", ""))

def ST_EDGE_FPOS(stack, buf):
    try:
        return (("ST_EDGE_FPOS", stack[-1].finePOS))
    except:
        return (("ST_EDGE_FPOS", ""))


def BUF_EDGE_POS(stack, buf):
    return (("BUF_EDGE_POS", buf[0].coarsePOS))

def BUF_EDGE_FPOS(stack, buf):
    return (("BUF_EDGE_FPOS", buf[0].finePOS))


def TOP_PAIR_W(stack, buf):
    try:
        return (("TOP_PAIR_W", (stack[-1].word, buf[0].word)))
    except:
        return (("TOP_PAIR_W", ("", buf[0].word)))


def TOP_PAIR_POS(stack, buf):
    try:
        return (("TOP_PAIR_POS", (stack[-1].coarsePOS, buf[0].coarsePOS)))
    except:
        return (("TOP_PAIR_POS", ("", buf[0].coarsePOS)))

def TOP_PAIR_FPOS(stack, buf):
    try:
        return (("TOP_PAIR_FPOS", (stack[-1].finePOS, buf[0].finePOS)))
    except:
        return (("TOP_PAIR_FPOS", ("", buf[0].finePOS)))


def ST_TOP_WPOS(stack, buf):
    try:
        return (("ST_TOP_WPOS", (stack[-1].word, stack[-1].coarsePOS)))
    except:
        return (("ST_TOP_WPOS", ("", "")))

def TOP_PAIR_WPOS(stack, buf):
    try:
        return (("TOP_PAIR_WPOS", (stack[-1].word, stack[-1].coarsePOS, buf[0].word, buf[0].coarsePOS)))
    except:
        return (("TOP_PAIR_WPOS", ("", "", buf[0].word, buf[0].coarsePOS)))

def TOP_PAIR_WPOS2(stack, buf):
    try:
        return (("TOP_PAIR_WPOS2", (stack[-1].word, stack[-1].coarsePOS, buf[0].word)))
    except:
        return (("TOP_PAIR_WPOS2", ("", "", buf[0].word)))

def TOP_PAIR_WPOS3(stack, buf):
    try:
        return (("TOP_PAIR_WPOS3", (stack[-1].word, buf[0].word, buf[0].coarsePOS)))
    except:
        return (("TOP_PAIR_WPOS3", ("", buf[0].word, buf[0].coarsePOS)))

def TOP_PAIR_WPOS4(stack, buf):
    try:
        return (("TOP_PAIR_WPOS4", (stack[-1].word, stack[-1].coarsePOS, buf[0].coarsePOS)))
    except:
        return (("TOP_PAIR_WPOS4", ("", "", buf[0].coarsePOS)))

def TOP_PAIR_WPOS5(stack, buf):
    try:
        return (("TOP_PAIR_WPOS5", (stack[-1].coarsePOS, buf[0].word, buf[0].coarsePOS)))
    except:
        return (("TOP_PAIR_WPOS5", ("", buf[0].word, buf[0].coarsePOS)))

def TOP_PAIR_BUFF_POS(stack, buf):
    try:
        return (("TOP_PAIR_BUFF_POS", (buf[0].coarsePOS, buf[1].coarsePOS)))
    except:
        return (("TOP_PAIR_BUFF_POS", (buf[0].coarsePOS, "")))

def TOP_TRIPLE_WPOS(stack, buf):
    try:
        return (("TOP_TRIPLE_WPOS", (stack[-1].coarsePOS, buf[0].coarsePOS, buf[1].coarsePOS)))
    except:
        if (stack == [] and len(buf) > 1):
            return (("TOP_TRIPLE_WPOS", ("", buf[0].coarsePOS, buf[1].coarsePOS)))
        elif (stack == [] and len(buf) <= 1):
            return (("TOP_TRIPLE_WPOS", ("", buf[0].coarsePOS, "")))
        else:
            return (("TOP_TRIPLE_WPOS", (stack[-1].coarsePOS, buf[0].coarsePOS, "")))

def TOP_TRIPLE_WPOS_head(stack, buf):
    try:
        h = ""
        for s in stack:
            if stack[-1].head == s.index:
                h = s
        for b in buf:
            if (stack[-1].head == b.index):
                h = b
        return (("TOP_TRIPLE_WPOS_head", (h.coarsePOS, stack[-1].coarsePOS, buf[0].coarsePOS)))
    except:
        return (("TOP_TRIPLE_WPOS_head", ("", "", buf[0].coarsePOS)))


# Test different combinations of feature weighting:
flist = [(TOP_PAIR_W, 1.0), (TOP_PAIR_POS, 1.0)
    , (BUF_EDGE_FPOS, 1.5), (TOP_PAIR_WPOS, 1.5), (TOP_PAIR_WPOS2, 1.0), (TOP_PAIR_WPOS3, 1.0),
          (TOP_PAIR_WPOS5, 1.0), (TOP_TRIPLE_WPOS, 1.0), (TOP_PAIR_BUFF_POS, 1.0), (TOP_PAIR_WPOS4, 1.0), (ST_TOP_WPOS, 1.0),
          (BUF_EDGE_POS, 1.0), (TOP_TRIPLE_WPOS_head, 1.5), (ST_EDGE_FPOS, 1.0), (TOP_PAIR_FPOS, 1.0)]

# Features not used
# (BUF_EDGE_W, 0.5), (ST_EDGE_POS, 0.5), (ST_EDGE_W, 1.0)


from collections import OrderedDict
import random


class Parser:
    def __init__(self, flist):
        # configuration
        self.stack = []
        self.buffer = []
        self.arcs = []

        # oracle - dict of dicts: {sent_id:{(stack, buf, arcs): transition}}
        self.oracle = {}
        # keeps configurations and parser's decision. Same structure as oracle
        self.trace = {}
        # parsed arcs - keeps only arcs for completed parses for easier extraction. Structure:
        # {sent_id: {arc_tail_index: arc_head_index}}
        self.parsed_arcs = {}

        self.weights = {"LARC": OrderedDict({}), "RARC": OrderedDict({}), "SHIFT": OrderedDict({}), "REDUCE": OrderedDict({})}
        self.cumulative_weights = {"LARC": OrderedDict({}), "RARC": OrderedDict({}), "SHIFT": OrderedDict({}), "REDUCE": OrderedDict({})}  # for averaging
        self.cumulative_counts = {"LARC": OrderedDict({}), "RARC": OrderedDict({}), "SHIFT": OrderedDict({}),
                                  "REDUCE": OrderedDict({})}

        self.transitions = ["LARC", "RARC", "SHIFT", "REDUCE"]
        self.flist = [f[0] for f in flist]  # list of features (more specifically, list of feature function names)
        self.f_scaler = {str(k.__name__): v for (k, v) in flist}
        self.root = Entry("0", "ROOT", "_", "ROOT", "ROOT", "_", "_", "_", "_", "_")

        self.iterBeginRandomChoice = 2
        self.randomChoiceProbability = 0.8

    # ==== Oracle functions ====

    def _determineNextTransition(self):
        """
        For training data, determine next transition (used to construct the oracle).
        """
        # Retrieve a list of heads which are still in the buffer not counting
        # the rightmost element of the buffer
        active_heads = [entry.head for entry in self.buffer]
        active_tails = [ent.index for ent in self.buffer]
        # If the stack is empty and the rightmost element of the buffer is ROOT,
        # we are drawing right arcs from the ROOT. We need to shift ROOT back to
        # the stack
        if self.stack == []:
            return ("SHIFT")
        # If the index of the head of the rightmost element of the stack
        # coincides with the index of the leftmost element of the buffer,
        # we can draw a left-arc
        if self.stack[-1].head == self.buffer[0].index:
            return ("LARC")
        # If the index of the righmost element of the stack coincides
        # with the index of the head of the rightmost element of the buffer
        # we can draw right-arc
        if (self.stack[-1].index == self.buffer[0].head):
            return ("RARC")
        # If the righmost element of the stack doesn't serve as head
        # at some later point in the parse, we can reduce
        if ((self.stack[-1].index not in active_heads) and (self.stack[-1].head not in active_tails)):
            return ("REDUCE")
        # If neither of the previous conditions are true, we can only shift
        return ("SHIFT")

    def _determineNextTransitions(self):
        """
        For training data, determine next transition (used to construct the oracle).
        For Dynamic Oracle
        """
        # Retrieve a list of heads which are still in the buffer not counting
        # the rightmost element of the buffer
        active_heads = [entry.head for entry in self.buffer]
        active_tails = [ent.index for ent in self.buffer]
        # If the stack is empty and the rightmost element of the buffer is ROOT,
        # we are drawing right arcs from the ROOT. We need to shift ROOT back to
        # the stack
        if self.stack == []:
            return (["SHIFT"])
        # If the index of the head of the rightmost element of the stack
        # coincides with the index of the leftmost element of the buffer,
        # we can draw a left-arc
        if self.stack[-1].head == self.buffer[0].index:
            return (["LARC"])
        # If the index of the righmost element of the stack coincides
        # with the index of the head of the rightmost element of the buffer
        # we can draw right-arc
        if (self.stack[-1].index == self.buffer[0].head):
            return (["RARC"])
        # If the righmost element of the stack doesn't serve as head
        # at some later point in the parse, we can reduce or shift
        # For dynamic parser ambiguity is only in the case of REDUCE
        if ((self.stack[-1].index not in active_heads) and (self.stack[-1].head not in active_tails)):
            return (["REDUCE", "SHIFT"])
        # If neither of the previous conditions are true, we can only shift
        return (["SHIFT"])

    def _constructGoldSeqOfTransitions(self, sent, sent_id, rnd_weights=True):
        """
        Given a parsed sentence, determine the sequence of transitions.
        :param sent
        :param sent_id - index of sentence in the text; used as key in self.oracle
                         for all configurations for this sentence
        :param rnd_weights: bool whether to initialize weights randomly. If True,
                    weights are selected randomly from -1,0,1; if False, weights are
                    all 0
        """
        # Initialize start configuration: stack only has ROOT in it
        self.stack = [self.root]
        # The buffer has the whole sentence in it. A COPY of the whole sentence ([:] piece)
        self.buffer = sent[:]
        self.arcs = []
        # Initialize oracle for the current sentence. The oracle is a dict with configurations
        # as keys and transitions as values. It's an OrderedDict so that we can later print the
        # sequence of transitions for debugging
        self.oracle[sent_id] = OrderedDict({})
        # Construct gold sentence from annotated data
        while self.buffer != []:
            next_transition = self._determineNextTransition()
            self.oracle[sent_id][(tuple(self.stack), tuple(self.buffer), tuple(self.arcs))] = next_transition
            # add features of the current configuration to the overall set of features
            self._initWeights(rnd_weights)
            self._applyTransition(next_transition)

    def constructOracle(self, sents, verbose=False, rnd_weights=True):
        for sent_id, sent in enumerate(sents):
            self._constructGoldSeqOfTransitions(sent, sent_id, rnd_weights)
            # give some indication of how we are doing
            if verbose:
                print("Constructing Oracle... processing sentence {0} out of {1}".format(sent_id + 1, len(sents)))


    def printGoldSeq(self, sent_id):
        """
        A function to print the recovered sentence of transition.
        :param sent_id - index of the sentence in the training data to return
        gold sequence of transitions for
        """
        gold_parse = self.oracle[sent_id]
        gold_seq = []
        for config, transition in gold_parse.iteritems():
            stack_words = [entry.word for entry in config[0]]
            buffer_words = [entry.word for entry in config[1]]
            gold_seq.append((stack_words, buffer_words, transition))
        for c in gold_seq:
            print(c, sep="\n")

    # ==== Trainer functions ====

    def _initWeights(self, rnd=True):
        """
        Function to add current feature combination to the dictionary of weights.
        Is called from within _constructGoldSeqOfTrans: in that function we are
        already iterating over all possible configurations, no need to do it again later
        just for weights initialization.
        """
        fs = self._extractFeatures()
        # Not used anymore
        # initialization now happens online in _updateWeights



    def _updateWeights(self, predicted_trans, gold_trans, fs):
        for f in fs:
            # NEW: multiply by feature importance weights (self.f_scaler)
            if (f not in self.weights[predicted_trans] or f not in self.weights[gold_trans]):
                for trans in self.transitions:
                    self.weights[trans][f] = 0
                    self.cumulative_weights[trans][f] = 0
                    self.cumulative_counts[trans][f] = 1
            self.weights[predicted_trans][f] -= 1 * self.f_scaler[str(f[0])]
            self.weights[gold_trans][f] += 1 * self.f_scaler[str(f[0])]

            # track cumulative weights for averaging them later
            self.cumulative_weights[predicted_trans][f] += self.weights[predicted_trans][f]
            self.cumulative_weights[gold_trans][f] += self.weights[gold_trans][f]

            self.cumulative_counts[predicted_trans][f] += 1
            self.cumulative_counts[gold_trans][f] += 1

    def train(self, sents, n_iter, shuffle=True, verbose=True):
        # for averaging
        updates_counter = 0
        for ii in range(0, n_iter):

            if verbose:
                print("Training the classifier... iteration {0} out of {1}".format(ii + 1, n_iter))

            # presentation sequence
            pres_seq = range(0, len(sents))
            # randomize, if necessary:
            if shuffle:
                random.shuffle(pres_seq)

            for i, sent_id in enumerate(pres_seq):
                if verbose:
                    print(" ---- Processing sentence {0} out of {1}".format(i + 1, len(pres_seq)))
                # initialize a start configuration
                self.stack = [self.root]
                self.buffer = sents[sent_id][:]  # notice that we create a copy of the sentence (by using [:])
                self.arcs = []

                while self.buffer != []:
                    # get features of the current configuration
                    fs = self._extractFeatures()
                    predicted_trans = self._predictTransition(fs, False)
                    gold_trans = self.oracle[sent_id][(tuple(self.stack), tuple(self.buffer), tuple(self.arcs))]
                    if predicted_trans != gold_trans:
                        self._updateWeights(predicted_trans, gold_trans, fs)
                        updates_counter += 1
                    self._applyTransition(gold_trans)

        # at the end of training, average all weights
        for trans in self.weights.iterkeys():
            for f in self.weights[trans].iterkeys():
                self.weights[trans][f] = self.cumulative_weights[trans][f] / float(self.cumulative_counts[trans][f])
                # Memory management
        self.cumulative_weights.clear()
        self.cumulative_counts.clear()
        return (self.weights)

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------       Dynamic Oracle implementation       -------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def transitionCost(self, trans):
        cost = 0
        active_heads = [entry.head for entry in self.buffer]
        active_tails = [ent.index for ent in self.buffer]
        inactive_heads = [entry.head for entry in self.stack]
        inactive_tails = [entry.index for entry in self.stack]

        if (self.stack == []):
            if (trans == "SHIFT"):
                return 0
            else:
                return 100

        s = self.stack[-1]
        b = self.buffer[0]
        currentArcTails = [el.tail_idx for el in self.arcs]

        if (trans == "LARC"):
            if (s.head == b.index):
                return 0
            # Case of the wrong move in the previous turn
            if (s.head != b.index and s.head not in active_tails and s.index not in active_heads):
                return 0

            if (s.index in currentArcTails):
                cost += 1

            for h in active_heads:
                if (s.index == h):
                    cost += 1
            for t in active_tails:
                if (s.head == t):
                    cost += 1
        elif (trans == "RARC"):
            if (s.index == b.head):
                return 0
            if (s.index != b.head and b.head not in inactive_tails and b.head not in active_tails and b.index not in inactive_heads):
                return 0

            for h in inactive_heads:
                if (b.index == h):
                    cost += 1
            for t in inactive_tails:
                if (b.head == t):
                    cost += 1
            for t in active_tails:
                if (b.head == t):
                    cost += 1
        elif (trans == "REDUCE"):
            for h in active_heads:
                if (s.index == h):
                    cost += 1
            for t in active_tails:
                if (s.head == t):
                    cost += 1
        else:  # SHIFT
            for h in inactive_heads:
                if (b.index == h and h not in currentArcTails):
                    cost += 1
            for t in inactive_tails:
                if (b.head == t):
                    cost += 1

        return cost

    def oracleSuccess(self, trans):
        if (self.transitionCost(trans) == 0):
            return True
        return False


    def trainDynamic(self, sents, n_iter, shuffle=True, verbose=True):
        # for averaging
        updates_counter = 0

        for ii in range(0, n_iter):

            if verbose:
                print("Training the classifier... iteration {0} out of {1}".format(ii + 1, n_iter))

            # presentation sequence
            pres_seq = range(0, len(sents))
            # randomize, if necessary:
            if shuffle:
                random.shuffle(pres_seq)

            for i, sent_id in enumerate(pres_seq):
                if verbose:
                    print(" ---- Processing sentence {0} out of {1}".format(i + 1, len(pres_seq)))
                # initialize a start configuration
                self.stack = [self.root]
                self.buffer = sents[sent_id][:]  # notice that we create a copy of the sentence (by using [:])
                self.arcs = []

                while self.buffer != []:
                    # get features of the current configuration
                    fs = self._extractFeatures()
                    predicted_trans = self._predictTransition(fs, False)
                    zero_cost_transitions = []
                    # construct set of zero cost transitions from current state
                    min_cost_transition = "SHIFT"
                    min_cost = 10000
                    for transition in self._determineNextTransitions():
                        transCost = self.transitionCost(transition)
                        if (transCost == 0):
                            zero_cost_transitions += [transition]
                            min_cost = 0
                        else:
                            if (transCost < min_cost):
                                min_cost = transCost
                                min_cost_transition = transition
                            elif (transCost == min_cost):
                                min_cost_transition = random.choice([transition, min_cost_transition])

                    if (zero_cost_transitions != []):
                        oracle_transition = self._computeArgMaxTrans(fs, zero_cost_transitions)
                    else:
                        if verbose:
                            print ("no zero-cost transitions found --> Going Random")
                        oracle_transition = min_cost_transition
                    if (self.stack == []):
                        oracle_transition = "SHIFT"


                    if (predicted_trans not in zero_cost_transitions and zero_cost_transitions != []):
                        self._updateWeights(predicted_trans, oracle_transition, fs)

                        updates_counter += 1

                    if (zero_cost_transitions != []):
                        chosen_transition = self._chooseTransitionEXP(ii, predicted_trans, zero_cost_transitions)
                    else:
                        chosen_transition = oracle_transition
                    if (self.stack == []):
                        chosen_transition = "SHIFT"
                    self._applyTransition(chosen_transition)

        # at the end of training, average all weights
        for trans in self.weights.iterkeys():
            for f in self.weights[trans].iterkeys():
                self.weights[trans][f] = self.weights[trans][f] / float(self.cumulative_counts[trans][f])
        self.cumulative_weights.clear()
        self.cumulative_counts.clear()
        return (self.weights)

    def _chooseTransitionAMB(self, currentIter, predictedTransition, zeroCostTransitions):
        if (predictedTransition in zeroCostTransitions):
            return predictedTransition
        else:
            return random.choice(zeroCostTransitions)

    def _chooseTransitionEXP(self, currentIter, predictedTransition, zeroCostTransitions):
        if (currentIter > self.iterBeginRandomChoice and random.random() > self.randomChoiceProbability):
            return predictedTransition
        else:
            return self._chooseTransitionAMB(currentIter, predictedTransition, zeroCostTransitions)

    # ==== Parser functions ====

    def _applyTransition(self, transition):
        if transition == "LARC":
            # get arc properties
            arc_head_idx = self.buffer[0].index
            arc_head_word = self.buffer[0].word
            arc_tail_idx = self.stack[-1].index
            arc_tail_word = self.stack[-1].word
            # save the new arc
            self.arcs.append(Arc(arc_head_idx, arc_tail_idx, arc_head_word, arc_tail_word))
            # remove the rightmost element from the stack
            self.stack.pop()
        elif transition == "RARC":
            # get arc properties
            arc_head_idx = self.stack[-1].index
            arc_head_word = self.stack[-1].word
            arc_tail_idx = self.buffer[0].index
            arc_tail_word = self.buffer[0].word
            # save the new arc
            self.arcs.append(Arc(arc_head_idx, arc_tail_idx, arc_head_word, arc_tail_word))
            # remove the leftmost element from the buffer
            self.stack.append(self.buffer.pop(0))
        elif transition == "SHIFT":
            # add the leftmost element of the buffer to the end of the stack
            self.stack.append(self.buffer.pop(0))
        elif transition == "REDUCE":
            # pop the righmost element of stack
            self.stack.pop()


    def _extractFeatures(self, stack=None, buf=None):
        """
        Given a set of feature functions in self.flist, apply them to the
        current configuration. One can specify a configuration; if not,
        it's taken from the current configuration the Trainer has
        """
        if stack == None:
            stack = self.stack
        if buf == None:
            buf = self.buffer
        out = []
        for f in self.flist:
            out.append(f(stack, buf))
        return (out)

    def _predictTransition(self, fs, filterReduce):
        """
        Predict next transition based on the features of current configuration (fs)
        """

        prediction = {}
        for trans in self.transitions:
            if (filterReduce and trans == "REDUCE"):
                continue
            else:
                prediction[trans] = self._computeTransitionValue(fs, trans)
                #for f in fs:
                #    try:
                #        prediction[trans] += self.weights[trans][f]
                #   # if an error happens (e.g. because the feature is not defined), assume that the weight is 0
                #    except:
                #        prediction[trans] += 0
        # Find out which prediction has the biggest score.
        # we may not worry about ties: if the incorrect and correct prediction are tied,
        # and the incorrect one is chosen, it will be adjusted, which is what we want anyways
        return (max(prediction.iterkeys(), key=(lambda key: prediction[key])))

    def _computeTransitionValue(self, fs, trans):
        """
        Predict next transition based on the features of current configuration (fs)
        """

        transValue = 0
        for f in fs:
            try:
                transValue += self.weights[trans][f]
            # if an error happens (e.g. because the feature is not defined), assume that the weight is 0
            except:
                transValue += 0
        # Find out which prediction has the biggest score.
        # we may not worry about ties: if the incorrect and correct prediction are tied,
        # and the incorrect one is chosen, it will be adjusted, which is what we want anyways
        return (transValue)

    def _computeArgMaxTrans(self, fs, transitions):
        prediction = {}
        for trans in transitions:
            prediction[trans] = self._computeTransitionValue(fs, trans)

        return (max(prediction.iterkeys(), key=(lambda key: prediction[key])))



    def parse(self, sents, verbose=True):

        for sent_id, sent in enumerate(sents):
            if verbose:
                print("Parsing... sentence {0} out of {1}".format(sent_id + 1, len(sents)))
            # init configuration
            self.stack = [self.root]
            self.buffer = sent[:]
            self.arcs = []

            # init parser output variables
            self.trace[sent_id] = OrderedDict({})
            self.parsed_arcs[sent_id] = {}

            while self.buffer != []:
                fs = self._extractFeatures()

                flReduce = False
                next_trans = self._predictTransition(fs, flReduce)

                # log parser's decisions
                if (self.stack == []):
                    next_trans = "SHIFT"
                self.trace[sent_id][(tuple(self.stack), tuple(self.buffer), tuple(self.arcs))] = next_trans
                self._applyTransition(next_trans)

            # if the tree was not projective, the stack will have more words than just root.
            # we'll connect them to the root to at least have a connected tree; mark this with
            # a special kind of "transition" - FIX_ROOT
            while len(self.stack) > 1:
                self.arcs.append(Arc(self.root.index, self.stack[-1].index, self.root.word, self.stack[-1].word))
                self.stack.pop()
                self.trace[sent_id][(tuple(self.stack), tuple(self.buffer), tuple(self.arcs))] = "FIX_ROOT"

                # create a dictionary of the kind {arc_tail: arc head} for easy extraction of
            # dependency info (trace contains this info, but it's too clumsy to dig in later)
            for arc in self.arcs:
                self.parsed_arcs[sent_id][arc.tail_idx] = arc.head_idx

    def printTrace(self, parse_trace, file_name):
        """
        Output parse trace information to a file.
        """
        with open(file_name, "w") as f_handle:
            for sent_id in parse_trace.iterkeys():
                for config, trans in parse_trace[sent_id].iteritems():
                    stack_words = [entry.word for entry in config[0]]
                    buffer_words = [entry.word for entry in config[1]]
                    arcs = [(arc.head, arc.tail) for arc in config[2]]
                    print("Stack: {0}\nBuffer: {1}\nArcs: {2}\nNext transition: {3}\n".format(str(stack_words),
                                                                                              str(buffer_words),
                                                                                              str(arcs), trans),
                          file=f_handle)
                print("=========================\n\n", file=f_handle)

    def outputParse(self, sents, parse_trace, file_name):
        """
        Output parsed sentence in the format suitable for depeval.py script
        (basically, fill in the heads in the test file)
        """
        with open(file_name, "w") as f_handle:
            for sent_id in parse_trace.iterkeys():
                for word in sents[sent_id]:
                    output_line = list(word)
                    try:
                        output_line[6] = self.parsed_arcs[sent_id][word.index]  # 6th field is the word head.
                    except: # In case the word got the transition REDUCE before appearing in the Arc
                        output_line[6] = 0
                    for field in output_line:
                        f_handle.write((str(field) + "\t"))
                    f_handle.write("\n")
                f_handle.write("\n")

    # ==== Helper functions ====

    def _constructTree(self, sent, parent):
        """
        Construct a list representation of the dependency tree.
        :param sent - list of elementrs of type Entry, representing the sentence
        :param parent - should be representation of ROOT:
            Entry("0", "ROOT", "_", "ROOT", "ROOT", "_", "_", "_", "_", "_")

        The output is of the following form: [ROOT [Head [Dep Dep]]] etc. I.e the first
        element is the head and the rest are dependents, each of which can have the same
        internal structure
        """
        # get the set of all heads in the sentence
        heads = set([entry.head for entry in sent])
        # initialize the output variable
        out = []
        # list of words which depend on the current head (ROOT on the first pass)
        current_dependents = [w for w in sent if w.head == parent.index]
        # for each dependent
        for dep in current_dependents:
            # add it to the outpur
            out.append(dep)
            # if dependent has dependents of its own, call the function recursively
            if dep.index in heads:
                out.append(self._constructTree(sent, dep))
        return (out)

    def _printTreePrivate(self, tree, level):
        """
        Prints tree made by _constructTree.
        :param tree
        :param level - you should pass 0 when you call it
        """
        # This line is useful in recursive calls: each time we make one, we need
        # to increase the level of embedding
        level = level + 1
        # For all entries in the list representation of the tree:
        for i, entry in enumerate(tree):
            # if it's not a list, it must be a leaf
            if type(entry) == Entry:
                # we tabulate it N levels to the right and decorate output a little bit
                tree_stem = "|"
                for l in range(1, level):
                    tree_stem = tree_stem + "\t" + "|"
                print((tree_stem + "---" + entry.word).expandtabs(6))
                # if we reached the last element at the current level
                if i == len(tree):
                    # decrease our level of tabulations
                    level = level - 1
            # if the element is a list, it's a subtree, call the function recursively
            elif type(entry) == list:
                # level = level+1
                self._printTreePrivate(entry, level)

    def printTree(self, sent):
        """
        Public method for printing the tree.
        :param sent - actual setnence representation (list of Entries), not index
        """
        tree = self._constructTree(sent, self.root)
        self._printTreePrivate(tree, level=0)

    def _isProjective(self, sent):
        """
        Determines whether a given tree is projective or not. Returns True for projective trees
        and False for non-projective
        """
        for word in sent:
            # the entry information is a set of strings, but we will need numbers. Convert
            word_index = int(word.index)
            word_head = int(word.head)
            # if head is to the right of the current word
            if word_index < word_head:
                # find out which words are to the right of the current word and to the left of its head
                # (i.e. in between the word and the head)
                span = [entry for entry in sent if (int(entry.index) > word_index and int(entry.index) < word_head)]
                # for each word in between:
                for s in span:
                    # if its head is farther to the left than the current word or farther to the
                    # right than the head of the current word, the arc will cross other arcs
                    if int(s.head) < word_index or int(s.head) > word_head:
                        return (False)
            # same kind of logic for the case when the head is to the left of the current word
            elif word_index > word_head:
                span = [entry for entry in sent if int(entry.index) < word_index and int(entry.index) > word_head]
                for s in span:
                    if int(s.head) > word_index or int(s.head) < word_head:
                        return (False)

        # if neither of the previous conditions triggered, the tree is projective
        return (True)


def readData(file_name):
    """
    The structure of the output: for each sentence, a list of Entry namedtuples,
    one per word (see definition of Entry in the beginning of the script)
    """
    with open(file_name, "r") as f_hndl:
        sents = f_hndl.read()
    # get sentences by splitting the whole text by double newline
    sents = sents.split("\n\n")
    # remove the last sentence - it will be an empty line, since the text files
    # end with a double new line
    sents = sents[0:-1]
    # for each sentence:
    #   split sentence by newlines - this will give us individual entries
    #   for each individual entry:
    #       split it by tabs - this will give us individual fields
    #       unpack those fields using * operator
    #       convert it to a named tuple of the type defined in the beginning of the script
    sents = [[Entry(*entry.split("\t")) for entry in sent.split("\n")] for sent in sents]
    return (sents)


import os
import sys

if __name__ == "__main__":

    if len(sys.argv) < 4:
        sys.exit(
            "Too few input arguments! The script requires 3 file names as arguments: train_texts, test_texts, parser_output")

    (train_file, test_file, output_file) = sys.argv[1:]

    # ==== Read texts in ====
    train_sents = readData(train_file)
    test_sents = readData(test_file)

    # ==== Parser operations ====

    # Define features we want to use.
    # Notice that now we pass *tuples*: feature function name AND weight we want to assign to this feature


    parser = Parser(flist)
    parser.constructOracle(train_sents, verbose=True,
                           rnd_weights=False)  # for tests, we set rnd_weights to False (all weights start with 0)



    # Train using Static Oracle
    parser.train(train_sents, shuffle=False, n_iter=5, verbose=True)

    #Train using Dynamic Oracle
    #parser.trainDynamic(train_sents, shuffle=False, n_iter=5, verbose=True)

    parser.parse(test_sents, verbose=True)


    print("Writing out results...")
    # parser.printTrace(parser.trace, "fancy_trace.txt")
    parser.outputParse(test_sents, parser.trace, output_file)

    # Parsing the dev set ---> Uncomment to use
    #parser.trace.clear()
    #dev_sents = readData("en.dev")
    #parser.parse(dev_sents, verbose=True)
    #parser.outputParse(dev_sents, parser.trace, "devoutput.txt")


    #os.system("python depeval.py en.dev fancy_parse.txt >> f_scaler_test.txt")
    print("Done!")

# ===========================================================
