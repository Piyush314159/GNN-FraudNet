# Graph-Based Fraud Patterns

## Fan-Out Pattern

A fan-out pattern occurs when a single node sends transactions to many different nodes. In the context of fraud detection:

- **High out-degree**: The source node has significantly more outgoing edges than average.
- **Fraud signal**: When combined with illicit neighbors, fan-out suggests fund dispersal or mixing.
- **Legitimate alternative**: Payment processors and exchanges also show high fan-out.
- **GNN detection**: GraphSAGE aggregates neighbor features, so a fan-out node with many illicit neighbors will inherit a strong fraud signal through message passing.

## Fan-In Pattern (Consolidation)

A fan-in pattern is the reverse — many nodes send to a single destination:

- **High in-degree**: The destination node receives from many sources.
- **Fraud signal**: Consolidation of illicit funds before exchange cash-out.
- **Legitimate alternative**: Merchant addresses and donation wallets show fan-in.
- **Key distinction**: The label distribution of source nodes matters more than the pattern itself.

## Dense Cluster

A dense cluster is a group of nodes with many interconnections:

- **High clustering coefficient**: Nodes in the group are heavily interconnected.
- **Fraud signal**: Coordinated fraud rings create dense transaction clusters.
- **Isolation**: Dense clusters that are weakly connected to the main graph are more suspicious.
- **GNN advantage**: Message passing through dense clusters amplifies signals — if several nodes in the cluster are illicit, the GNN will propagate this evidence to all cluster members.

## Chain Pattern (Peeling)

A chain pattern involves sequential transactions through a series of addresses:

- **Linear topology**: Each node connects to at most one predecessor and one successor.
- **Fraud signal**: Layering technique to obscure fund origin.
- **Path length**: Longer chains indicate more deliberate obfuscation.
- **Diminishing amounts**: Each hop may peel off a small amount, with the bulk continuing.

## Star Pattern

A star pattern has one central node connected to many peripheral nodes:

- **Central hub**: One node has high degree while connected nodes have low degree.
- **Fraud signal**: Can indicate a mixing service, exchange, or coordinated scheme.
- **Hub label**: The label of the central node strongly influences GNN predictions for all spokes.
- **Asymmetric flow**: If flow is predominantly inward or outward, it adds directional context.

## Illicit Neighbor Concentration

The proportion of known-illicit neighbors is a strong fraud indicator:

- **High illicit ratio**: If more than 50% of a node's neighbors are labeled illicit, the node itself is very likely illicit.
- **Moderate ratio (20-50%)**: Warrants investigation — the node may be a legitimate entity interacting with illicit parties.
- **Low ratio (<20%)**: Lower risk, but context matters.
- **Unknown neighbors**: Many unknown-label neighbors add uncertainty. The model's prediction may be less reliable.

## Temporal Clustering

When examining the temporal features in the Elliptic dataset:

- **Same time step**: Transactions within the same time step are more likely to be related.
- **Burst activity**: Sudden increases in transaction volume from a node are suspicious.
- **Cross-step patterns**: Illicit activity often spans specific time periods, corresponding to active fraud campaigns.
- **The Elliptic dataset has 49 time steps**, each representing approximately two weeks of Bitcoin activity.
