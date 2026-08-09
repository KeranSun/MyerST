# Box 1 | Why spatial attribution needs a new axiom: from Shapley to Myerson

**The hidden assumption in Shapley-style explanations.** Attribution methods rooted in cooperative game theory — SHAP, and its graph-XAI descendants — assign each player (gene, spot, cell) a fair share of a model's output by averaging its marginal contribution over *all* coalitions:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\big[v(S \cup \{i\}) - v(S)\big]$$

Embedded in this formula is an assumption so natural it usually goes unstated: **any subset of players can form a coalition**. Two spots on opposite ends of a tissue section are treated as equally able to "cooperate" as two physically adjacent cells. In a tissue, this is false. Molecular cooperation — juxtacrine signaling, neighborhood composition effects, domain-boundary formation — is physically constrained by spatial adjacency. Evaluating marginal contributions over spatially impossible coalitions does not merely waste computation; it systematically distorts credit, because graph neural network hosts propagate information along exactly those adjacencies that free coalitions ignore.

**Communication games: coalitions restricted by a graph.** In 1977, Roger Myerson introduced cooperative games restricted by a communication graph — work later recognized by the 2007 Nobel Memorial Prize in Economics. Given a graph $g$ on players $N$, the worth of a coalition $S$ is redefined as the sum of the worths of its **connected components** in $g$:

$$v^g(S) = \sum_{C \in \Pi(S,\, g|_S)} v(C)$$

Disconnected fragments cannot pool their value. The *Myerson value* is the Shapley value of this induced game, and Myerson proved it is the **unique** allocation satisfying two axioms:

- **Component efficiency** — each connected component distributes exactly its own worth among its members: $\sum_{i \in C} \phi_i^M = v(C)$ for every component $C$.
- **Fairness** — a link between two players changes their payoffs *equally*: $\phi_i^M(v,g) - \phi_i^M(v,g - ij) = \phi_j^M(v,g) - \phi_j^M(v,g - ij)$.

**A three-spot example.** Consider three spots in a path $s_1 - s_2 - s_3$ with coalition worth $v(S) = |S|^2$. The Shapley value assigns every spot an identical share $(3, 3, 3)$ — free coalitions make the three spots perfectly exchangeable, and *position is invisible*. The Myerson value assigns $(\tfrac{8}{3}, \tfrac{11}{3}, \tfrac{8}{3})$: the middle spot earns 37% more than the endpoints because every spatially meaningful coalition must pass through it. Topology becomes visible in the credit assignment — precisely the property spatial omics attribution requires.

**What the framework buys in practice.**

1. **Topology-respecting credit.** Attributions are computed only over coalitions the tissue can physically realize (connected subgraphs of the spatial kNN graph).
2. **A built-in audit.** Component efficiency yields an exact identity, $\sum_i \phi_i = v(N) - v(\varnothing)$, that holds for the exact value and for every single Monte Carlo permutation. Any implementation or convergence failure shows up immediately as a violation — gradient- and perturbation-based explainers have no analogous self-check. We verify this identity to machine precision in all experiments.
3. **Principled edge attribution.** The fairness axiom equates bilateral link gains for both endpoints, giving a well-defined decomposition of node credit into edge contributions (first-order form: the pairwise synergy $v(\{i,j\}) - v(\{i\}) - v(\{j\}) + v(\varnothing)$, which we use for cell–cell communication edges).
4. **Computability.** Exact evaluation is #P-hard, but permutation sampling over connected coalitions converges at the standard Monte Carlo rate $O(M^{-1/2})$ (Extended Data Fig. X), and coalition evaluations are batched and cached. Attribution of a ~350-spot boundary band takes ~25 minutes on a single CPU.

**Scope and honest caveats.** (i) When players are nearly exchangeable — e.g., a gene and its highly correlated passenger copy — any Shapley-family value splits credit between them *by design*; this is correct behavior, and benchmarks must not penalize it. (ii) The value answers "how does the model use the tissue graph", not "what is biologically causal"; causal claims require the perturbation validations we pair with every explanation. (iii) For purely local questions (a single receiver cell's signaling inputs), small exact local games are preferable to global sampling — a distinction our framework exposes rather than hides (Section X).
