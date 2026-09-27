# Bitcoin Fraud Typologies

## Mixing Services (Tumblers)

Mixing services are designed to obscure the origin of Bitcoin by pooling funds from multiple users and redistributing them. Key characteristics:

- **Fan-out pattern**: One input address splits into many output addresses.
- **Rapid cycling**: Funds move through multiple intermediate addresses within short time windows.
- **Similar amounts**: Output transactions are often of similar value (splitting evenly).
- **Short-lived addresses**: Intermediate addresses are used once and abandoned.
- **High transaction frequency**: Many transactions occur in rapid succession from related addresses.

Mixing services are used by both legitimate privacy-conscious users and criminals attempting to launder illicit funds. Their presence in a transaction graph is a strong risk indicator but not conclusive proof of fraud.

## Ponzi Schemes

Ponzi schemes on Bitcoin follow a characteristic pattern:

- **Funnel pattern**: Many input addresses send funds to a central address.
- **Delayed redistribution**: Funds are held and redistributed over time to create the illusion of returns.
- **Decreasing returns**: Later participants receive smaller payouts.
- **Growing fan-in**: The number of input addresses grows over time as more victims invest.
- **Sudden collapse**: Transaction activity drops abruptly when the scheme collapses.

## Ransomware

Ransomware payments show distinct patterns:

- **Single large payments**: Victims send specific demanded amounts.
- **Rapid consolidation**: Multiple ransom payments are quickly consolidated into fewer addresses.
- **Exchange cash-out**: Consolidated funds are moved to exchange addresses for conversion to fiat.
- **Time clustering**: Payments cluster around the time of the ransomware attack.
- **Fixed amounts**: Multiple transactions with the same value suggest ransom demands.

## Dark Market Activity

Transactions associated with darknet marketplaces exhibit:

- **Regular small transactions**: Consistent purchase-sized payments.
- **Hub-and-spoke pattern**: Central market address connected to many buyer/seller addresses.
- **Escrow patterns**: Three-party transaction structures (buyer → escrow → seller).
- **Regular timing**: Transactions follow patterns matching marketplace operating hours.
- **Withdrawal clustering**: Sellers periodically withdraw accumulated funds.

## Money Laundering (Layering)

The layering stage of money laundering involves:

- **Chain transactions**: Funds pass through a chain of addresses in sequence.
- **Round-tripping**: Funds return to addresses near the origin through complex paths.
- **Cross-chain activity**: Funds move between different cryptocurrency networks.
- **Smurfing**: Large amounts are broken into many small transactions below reporting thresholds.
- **Nested services**: Use of multiple intermediary services to add layers of obfuscation.

## Phishing and Theft

Stolen funds exhibit patterns such as:

- **Sudden large transfer**: A large amount moves from a previously dormant address.
- **Rapid dispersal**: Stolen funds are immediately split across many addresses.
- **No return transactions**: Unlike legitimate commerce, stolen funds show one-way flow.
- **New address usage**: Funds move to freshly created addresses with no prior history.
