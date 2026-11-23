# Quorum

**When several insiders at one company buy in the same month, what happens next?**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

Executives must report their own trades. The research says clusters of buying predicted returns. That research is now well known. Does the effect still exist?

## What this will contain

- A loader for the SEC's quarterly insider transaction data sets
- Cleaning rules, written down
- Cluster detection: three or more insiders buying within a window
- The split between routine and opportunistic traders
- Event-study returns against the market with costs, before and after 2012

## The result it is built to show

One chart: the return after a cluster, before the research was published and after.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Lakonishok and Lee, "Are Insider Trades Informative?", Review of Financial Studies 14(1), 2001
- Cohen, Malloy and Pomorski, "Decoding Inside Information", Journal of Finance 67(3), 2012

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

The SEC's Insider Transactions Data Sets: free quarterly files built from Forms 3, 4 and 5. Prices come from a free source.

## How it is tested

The maths is tested before any result is trusted.

- Cluster detection on a hand-built set of filings
- The routine-trader rule against the paper's definition
- No return is measured before the filing was public

## Layout

```
src/quorum/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] A loader for the SEC's quarterly insider transaction data sets
- [ ] Cleaning rules, written down
- [ ] Cluster detection: three or more insiders buying within a window
- [ ] The split between routine and opportunistic traders
- [ ] Event-study returns against the market with costs, before and after 2012
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- Free price data leaves out companies that were delisted. That flatters any strategy that buys. This is the biggest weakness here and it is stated first.
- The original papers used commercial data over 1986 to 2007. This is a different source and a different period, so it is a test of the idea, not an exact replication.
- Filings are as filed. The SEC does not guarantee their accuracy.
- This is research code. It is not investment advice and it does not trade.

## The name

A quorum is the number of members who must be present before a vote counts. One insider buying is an anecdote.

## Licence

MIT. See `LICENSE`.
