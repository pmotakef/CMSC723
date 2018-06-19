# Description:

After implementing the transition parser using the arc-standard method the baseline accuracy for the dev set was 38%. We implemented couple of improvements to the code for part 3, which raised the accuracy on the dev set as high as 80.3%. Below is the description of what we did for such improvements.

1 - Implementation of new transition system: Arc-eager method:

To implement the new transition system, some methods needed to be redone. First of all, the allowed transitions are now 4 (compared to 3 in the case of arc-standard which are "Shift", "Left Arc", and "Right Arc"). Therefore, "Reduce" was added to the allowed transitions. 

Second the method which determines the next transition based on the current configuration ([stack, buffer, arc] tuple) was changed. The major changes here are for "Right Arc" which acts immediately and "Reduce" which removes the top of stack if this element does not have any dependent or head in the buffer.

The next major change is in applyTransition method. Here "Right Arc" changed to add the bottom of buffer to top of the stack and produce the right arc, and "Reduce" which pops the top of stack.

Since Reduce may result in popping the "ROOT" new condition was implemented in case of empty stack: only allowed transition is "Shift".

Effect:

Above sums up the major work for implementing Arc-eager transition. This implementation allowed for major boost in accuracy, in the early runs we gained about 6% increase in accuracy on the dev set. All later additions/implementations are on top of this transition.

2 - Dynamic Oracle implementation:

Dynamic oracle was implemented following the works of Goldberg and Nivre: (A Dynamic Oracle for Arc-Eager Dependency Parsing) and (Training Deterministic Parsers with Non-Deterministic Oracles). This implementation follows line by line of those articles.
First a transition cost calculator function implemented which computes number of arcs that are lost (not reachable anymore) due to a transition. The conditions of how these costs are computed follows page 965 of Goldberg and Nivre, (A Dynamic Oracle for Arc-Eager Dependency Parsing) article.

Then the trainer calls all possible transition from the current configuration and calculates the cost of each transition. Transitions with zero cost are added to the zero_cost_transitions array. Then oracle chooses the best path using the current weights of features available to it ( argmax Sum(w(i).f(i)) ). In case no zero cost path was found (due to erroneous previous step) the oracle chooses the minimum cost path  as the oracle choice. The weights are updated if the predicted path is not in the zero_cost_transition list given this list is non-empty.

In the last step trainer chooses the transition path by calling the _chooseTransitionEXP(self, currentIter, predictedTransition, zeroCostTransitions) method. When the currentIter is greater than 3 (or 2, can be adjusted) the method considers choosing the predicted transition based on random chance, whether it is a correct or incorrect path. Otherwise _chooseTransitionAMB is called which chooses the next path based on zero_cost. If the predicted path is already among zero_cost paths, predicted is chosen, otherwise, randomly one of the zero_cost paths are chosen. This breaks if the zero_cost_transition is empty. In such a case trainer chooses the min_cost_path.

Effect:

The effect of this implementation found to be minimal on the accuracy (also this minimal effect can be seen in both referencing articles). The accuracy changed within +- 1% on the dev set. Since there are random occurrences in this implementation, each run yields to different answer. Therefore, to optimize the code further, we didn't use the dynamic oracle for the rest of the project, but kept the implementation in the code. All scores from later parts are based on the static oracle.

3 - New features:

Most of the features implemented here are borrowed from Zhang and Nivre (Transition-base dependency parsing with rich non-local features). The features in this part have importance factor which tells how important this feature to gain better accuracy. This factor is multiplied to the +-1 weight updates in the perceptron ( newWeight(i,f) = oldWeight(i,f) +-1 * feature factor(f) )
Following are the feature names x importance factor, and their description. The feature importance factors are tuned for the max accuracy.

Single Word Features:

a) BUF_EDGE_FPOS x 1.5: Head of the buffer Fine POS (single word) : Not from the article.

b) ST_EDGE_FPOS x 1.0: Top of stack Fine POS (Single word) : Not from the article.

c) ST_TOP_WPOS x 1.0: Top of the stack coarse POS and word : From the article.

d) BUF_EDGE_POS x 1.0: Head of the buffer coarse POS (single word) : From part 2

Two Word Features

e) TOP_PAIR_W x 1.0: pair of words at top of stack and head of the buffer : From part 2

f) TOP_PAIR_POS x 1.0: pair of coarse POS at top of stack and head of buffer : From part 2

g) TOP_PAIR_FPOS x 1.0: Top of the stack and buffer head pair using their Fine POS : Not from the article.

h) TOP_PAIR_WPOS x 1.5: Top of the stack (word and coarse POS) and head of buffer (word and coarse POS) pair : From the article. 

i) TOP_PAIR_WPOS2 x 1.0: Top of the stack (word and coarse POS) and head of buffer (word) pair : From the article. 

j) TOP_PAIR_WPOS3 x 1.0: Top of the stack (word) and head of buffer (word and coarse POS) pair : From the article. 

k) TOP_PAIR_WPOS4 x 1.0: Top of the stack (word and coarse POS) and head of buffer (coarse POS) pair : From the article. 

l) TOP_PAIR_WPOS5 x 1.0: Top of the stack (coarse POS) and head of buffer (word and coarse POS) pair : From the article. 

m) TOP_PAIR_BUFF_POS x 1.0 : 1st and 2nd head of buffer coarse POS (buffer[0] and buffer[1]) : From the article.

Three Word Features

n) TOP_TRIPLE_WPOS x 1.0: coarse POS of the 3 words: 1 on the top of stack and 2 are the 1st and 2nd head of buffer (stack[-1], buffer[0], and buffer[1]) : From the article.

o) TOP_TRIPLE_WPOS_head x 1.5: course POS of the 3 words: 1) head of the top of the stack, 2) top of the stack, and 3) head of buffer ( POS(stack[-1].head), stack[-1], buffer[0]) : From the article.

Effect:

Implementation of new features boosted the accuracy more than any other implementation we did. With these new features, the accuracy on the dev set using just the small training set (tr100) increased to 71%. We achieved largest improvements by features relating 3 words. Only 2 of such features were implemented and each improved the accuracy by about 5%. Also, we found that the 2 word features have more significant effects than single word features. Actually, we found out that removing single word features implemented in part 2 (such as identity of word and POS at top of the stack, and identity of word and POS at bottom of the buffer cause improvement in accuracy (~2-3% in total), since these features are loosely representative of their heads, while 2 and 3 words features introduce greater bond between features that produce arcs. Fine tuning the importance factors resulted in ~2% improve in accuracy, showing POS based features are more important (have bigger effect) than word based features.
In total we gained about 30% improvement in accuracy for dev set using new features.

4 - Larger training set (en.tr):

The last part of the improvement in the accuracy of dev set is gained by using the larger training set. Since there are more occurrences of different configurations, the training is more efficient and as a result we gained ~10% improvement in the accuracy of the dev set.

Final thought:

The biggest impact for improving the accuracy comes from better feature implementation. Larger training set comes in the second place. The change in transition system can have a good impact, which puts it in the 3rd place. The dynamic oracle didn't have much effect in improving the accuracy and given how challenging it is to implement for marginal gain, we find it nor worth while for the given set.
