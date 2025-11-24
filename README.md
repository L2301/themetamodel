## Core Ideas and Motivation (GPT Cleaned Version)

We model the latent space of a transformer as a Voronoi tessellation where each cell corresponds to a token.
Within this space, certain low-dimensional manifolds carry the actual semantic dynamics of the model.
A token’s hidden representation lies on one such “meaningful manifold” and its location determines the
distribution over the next token.

A single attention head acts as a geometric operator:

1. Q/K projection learns a metric space that captures the intrinsic geometry of a manifold
(i.e., distances encode transition probabilities).

2. Attention computes distances in this learned space.

3. V projection maps those geometric relations onto a new manifold representing the next hidden state.

4. MLP refines this manifold-to-manifold mapping locally.

Thus one attention head is:

    a learned geometric morphism between manifolds representing successive steps in a Markov chain.

Experimental Setup

We construct a simple Markov chain *mkv_1* whose transition probabilities reflect Euclidean distances on a 
known 2D manifold (a torus). Each state corresponds to a token. Thus the transition matrix defines a graph 
metric, which in turn defines a ground-truth manifold *M_1*

Context creates higher-order transitions.

For example, a 2-token context defines a second-order Markov chain *mkv_2*, whose geometry corresponds to another manifold
​*M_2*. Under this view, layers of a transformer correspond to moving between manifolds induced by higher-order transition
structure.

To test this, we:

1. Generate sequences from the torus-Markov chain.

2. Train 1-layer, 2-layer, and 3-layer attention-only transformers.

3. For each model, measure:

4. whether Q/K recovers the torus geometry

5. whether attention recovers the correct transition probabilities

5. whether deeper layers learn higher-order chains

6. how the manifolds evolve across layers

We report:

- KL divergence between *P_true* & *P_model*

- correlation between true torus distances and Q/K distances

- Implied learned torus radii (major/minor)

- neighborhood overlap

- and learned manifolds across layers

- logit distribution for arbitrary vectors


# Core Ideas and Motivation (original)

Take all of latent space to be a vornoi map where each region is a token (polytopes). In this space 
there exist 'meaningful manifolds' wherein the geometry of the manifold encodes a markov chain of 
token probabilities. 

A single headed transformer block can be thought of as taking in vectors on one 'meaningful manifold' 
and then projecting into a space that represents geometric relationships on this manifold measuring 
distance and then using that distance to place the vectors on the next meaningful manifold

The is done by attention heads taking in vectors mapping them to the Q-K space which learns to
encode the intrinsic geometry of the 'meaningful manifold', then given the distance measurment 
the value projection learns to 'locate' these vectors on the next 'meaningful manifold' and the
MLP layer takes these new more contextual vectors and refines their positions on this manifold.

We demonstrate the above using a 1 layer, 2 layer, and 3 layer transformer, a specially constructed
data set, and measurements of the learned representations. 

## Experimental Setup

We define a markov chain *mkv_1* whose state transitions probabilities are taken to be eculidean^[*] distances
between states. With each state representing an individual token. These probabilities uniquely encode a manifold 
*M_1* wherein location on this manifold encodes *⎥h〉*(h is the hidden representation) as probability distribution of 
which token it represents. Context is represted in the form of a higher order markov chain *mkv_2* that encodes the
geometry of Manifold *M_2* wherein the distance on *M1* (transition probability in *mkv_1*) is mapped to a simpler markov 
chain *(mkv^1)_2* that encodes the probability of the next token in the sequence given that context. This is extended
for each additional token of context. So for a three token sequence the distances between tokens 1, 2 on Manifold
*M_2* are used to calculate a probability distribution over next possible tokens.^[1]

We use a markov chain who's probabilites can be represented as a torus, to generate a library of sequences. We then use
this library to train a 1 layer transformer an

--
^[*]: This is not necessary but useful for the experiment in general the probability distribution defines a 
quasimetric, and the manifold is a space that can be modeled as a topological surface in latent space. Also 
worth noting that there are some instances where this is non trivial such a the doubling of tokens in a sequence
when this occurs the second character is taken on the next temporal dimension/manifold to avoid violating basic 
notion of distance in metric spaces.

^[1]: It is assumed but not investiagetd here that in a multi-headed configuration the resulting manifolds are 
combinations of the manifolds used by each individual head. 

