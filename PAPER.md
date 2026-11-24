# Sources, formulas and comparison

## Citations

1. Lakonishok, J. and Lee, I. (2001). "Are Insider Trades Informative?" Review of Financial Studies 14(1), 79-111. I could not reach the journal text. I read the abstract of the working paper version (NBER Working Paper 6656, July 1998), which is a scanned PDF, and nothing else from it. No formula here is taken from this paper.
2. Cohen, L., Malloy, C. and Pomorski, L. (2012). "Decoding Inside Information." Journal of Finance 67(3), 1009-1043. I read the working paper version (NBER Working Paper 16454, October 2010). Page and table numbers below refer to that version and may differ from the journal.
3. Balogh, A. (2023). "Insider trading." Scientific Data 10. doi:10.1038/s41597-023-02147-6. The dataset is doi:10.7910/DVN/VH6GVH. The article page would not load without a sign-in redirect, so the table layout was read from the files themselves.
4. Barber, B. and Lyon, J. (1997). "Detecting long-run abnormal stock returns: The empirical power and specification of test statistics." Journal of Financial Economics 43(3), 341-372. Not fetched. The buy-and-hold abnormal return is written from memory of its equation 3.
5. Newey, W. and West, K. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica 55(3), 703-708. Not fetched. The estimator is the standard Bartlett-kernel form and is checked against a hand calculation in the tests.
6. Jensen, M. (1968). "The Performance of Mutual Funds in the Period 1945-1964." Journal of Finance 23(2), 389-416. Not fetched. Used only for the CAPM intercept.

## What is implemented

### Routine and opportunistic insiders

Source: Cohen, Malloy and Pomorski, section I (working paper pages 11 to 12) and the note to Table I.

The paper requires "at least one trade in each of the three preceding years" before an insider is classified, and defines a routine trader as one who "placed a trade in the same calendar month for at least three consecutive years". Everyone else who meets the first condition is opportunistic. Insiders are labelled at the start of each calendar year, and all their trades from then on take the label.

`quorum.routine` does this per insider and firm. For year Y it looks at years Y-1, Y-2 and Y-3. One addition: a trade counts only if its filing was public before 1 January of year Y, so the label never uses a late filing.

Differences from the paper:

- The paper identifies people with the data vendor's person codes. Here an insider is a reporting-owner CIK at an issuer CIK.
- The paper restricts to stocks with CRSP and Compustat data. There is no such filter here.

### Cluster events

This definition is mine. Neither paper defines a cluster event this way. Lakonishok and Lee's abstract says purchases are more informative than sales and that predictability comes from smaller firms; the idea that several buyers at once is a stronger signal is common in later work, and I did not verify which paper first tested it.

On a day `d` when a purchase becomes public, count the distinct insiders with a purchase public on or before `d` and traded within the 30 days before `d`. Three or more is an event on `d`, unless the issuer had an event in the previous 182 days. Only officers and directors count, and a filing counts as one insider however many reporting persons it lists.

An event also needs a median reported purchase price of at least $2, a ticker for the issuer's CIK, a close on the entry day, and a median daily dollar volume of at least $100,000 in the month before.

### Event-study returns

The position opens at the close of the first trading day after the acceptance date of the filing. With `V_i(h)` the value of one unit after `h` trading days:

```
V_i(h)    = prod_{s=1..h} (1 + r_{i, entry+s})
BHAR_i(h) = V_i(h) - V_b(h)
```

This is the buy-and-hold abnormal return of Barber and Lyon (1997), equation 3, from memory. The daily return is `(close_t + dividend_t) / close_{t-1} - 1` on split-adjusted closes. A day with no return counts as zero.

Four benchmarks `b` are used.

- SPY and IWM.
- An equal-weight buy-and-hold portfolio of every liquid stock in the price file, formed at the same close. The first write-up (commit 8e16d1d) used this as the headline because it shares the survivorship of the event stocks.
- Matched controls, added afterwards. For each event, take the stocks that are liquid in the entry month, have a return on the entry day, and had no cluster of their own within 182 calendar days either side. The five nearest in log dollar volume (the median daily dollar volume of the month before) form an equal-weight buy-and-hold portfolio. Dollar volume stands in for size because the data has no market capitalisations. Every event in the published run has five controls.

### Standard errors in event time

The first results clustered the standard error of the mean on the calendar month of entry:

```
se = sqrt( sum_g ( sum_{i in g} (x_i - mean) )^2 ) / n
```

That treats events in different months as independent. A 63-day window covers about three months, so an event entered in March shares most of its return days with one entered in April. The standard error is then too small.

The tables now report three standard errors for every cell:

- `se_month_cluster`: the original one, kept for comparison.
- `se_quarter_cluster`: the same formula with entry quarters as groups.
- `se_overlap`: scores are summed within entry months, `s_m = sum_{i in m} (x_i - mean)`, and months up to `L` apart may be correlated, with Bartlett weights:

```
var(mean) = [ sum_m s_m^2 + 2 sum_{j=1..L} (1 - j/(L+1)) sum_m s_m s_{m-j} ] / n^2
```

A month with no events has a zero score. `L` is twice the number of months the window spans: 2 for 21 days, 6 for 63, 12 for 126. Twice the span keeps a weight of more than one half on every lag where windows can overlap. I set that rule before computing anything with it and did not try another. The t-statistics quoted in the README are the `se_overlap` ones.

The change between periods is the coefficient on a post-July-2012 dummy in a regression of the abnormal return on a constant and the dummy, with the same three standard errors.

A simulation in the tests draws events with a true mean of zero whose values share shocks across three months. Month clustering rejects at the 5% level more than 20% of the time. The overlap estimator stays under 12%.

What the correction did to the first headline (63 days, all liquid stocks benchmark, before July 2012, mean 2.8%):

| Standard error | t |
|---|---|
| Clustered by entry month | 2.68 |
| Clustered by entry quarter | 2.42 |
| Overlap-robust, 6 lags | 2.43 |

### Medians

The median has no simple standard error here. Its interval comes from a moving-block bootstrap: whole months are drawn in blocks of six consecutive months, 2,000 draws, fixed seed, and the interval is the 2.5% and 97.5% percentiles. The interval for the change draws the two periods independently.

### Calendar-time portfolios

Source: Cohen, Malloy and Pomorski, Table IV note: a firm is in the portfolio "in month (t+1) if any of its insiders placed an opportunistic buy on the firm in month t", rebalanced monthly.

Here month t is the month the filing became public, portfolios are equal weighted, and cluster stocks are held for three months. Abnormal return is the excess over the equal-weight return of all liquid stocks, and separately the intercept of

```
R_p - R_f = alpha + beta (R_m - R_f) + e
```

with SPY as the market and the month's average 3-month Treasury yield over 12 as `R_f`. Standard errors are Newey-West with three lags:

```
S = G_0 + sum_{j=1..L} (1 - j/(L+1)) (G_j + G_j'),   G_j = sum_t u_t u_{t-j}'
```

Each month is one observation, so overlapping holding periods do not inflate this test. The cluster portfolio holds about 46 stocks a month before July 2012 and 59 after.

The paper also reports Fama-French, Carhart, DGTW and five-factor alphas and value-weighted portfolios. None of those are built here: there are no factor returns or market capitalisations in the data used.

### Survivorship bounds

For each period the inputs are the priced events with their mean abnormal return `m_p` (63 days, all liquid stocks benchmark), and the clusters above the price floor whose issuer has no prices in the file. The missing clusters are scaled by the share of in-file clusters that pass the liquidity screen, giving `n_u`. A scenario says a share `p` of the missing events lost `L` against the benchmark and the rest earned `m_o`, which is either `m_p` or zero:

```
M = ( n_p m_p + n_u ( p L + (1 - p) m_o ) ) / ( n_p + n_u )
```

| | Before July 2012 | From July 2012 |
|---|---|---|
| Clusters above the price floor | 7,549 | 7,234 |
| Issuer in the price file | 2,494 | 4,323 |
| Issuer not in the price file | 5,055 | 2,911 |
| Priced (`n_p`) | 1,706 | 3,205 |
| Share of in-file clusters priced | 68% | 74% |
| Missing, scaled (`n_u`) | 3,458 | 2,158 |
| `m_p` | 2.8% | 0.4% |

`M` is linear in `p`, so the break-even shares in the README are exact for these assumptions. The assumptions are generous in one way and harsh in another. Many companies leave the file through a takeover at a premium, which would raise `m_o`. And a company that fails years later may already be sliding in the three months after its insiders bought, which would lower it. I have no data on either.

### The check of the join to prices

A split dated `s` with factor `f` (new shares per old share) means every adjusted close before `s` was divided by `f`:

```
close_as_traded(t) = close_adjusted(t) * prod_{splits s > t} f_s
```

An event is reconciled when the median reported purchase price lies between the lowest and highest as-traded close of the 30 calendar days up to the event date, widened by 10% on each side. The range is used because a cluster's purchases are spread over that window, and prices moved a long way inside 30 days in 2008 and 2020. The 10% and the 30 days were set before the check was run. `results/mapping_check.csv` has every year.

## Claim by claim

| Claim in the source | Source number | Here | Match |
|---|---|---|---|
| Opportunistic buys minus routine buys, equal weight, average monthly return | 0.68% (1989-2007), Table IV panel A | 0.16% before July 2012 (t 0.5, 77 months); -0.08% after (t -0.3, 160 months); 0.00% overall | No |
| Opportunistic buys, equal weight, CAPM alpha | 1.51% a month (t 5.89) | 0.61% before (t 1.5); -0.55% after (t -1.8) | No |
| Routine buys, equal weight, CAPM alpha | 0.92% a month (t 4.34) | 0.48% before (t 1.4); -0.38% after (t -1.5) | No |
| Opportunistic buys minus opportunistic sells, equal weight | 1.55% a month average | 0.49% before (t 1.2); -0.46% after (t -2.0) | No |
| The classified sample is about one third of all insider transactions | section I | 15% of purchases before July 2012 and 21% after | Lower |
| Purchases are informative (Lakonishok and Lee abstract) | qualitative | Firms with any officer or director purchase beat the all-liquid-stocks benchmark by 0.19% a month before July 2012 (t 1.6) and 0.07% after (t 0.6) | Weak |
| Clusters of buyers predict returns | no source number | Among survivors, 2.5% over 63 days against matched controls before July 2012 (t 2.0) and 0.2% after (t 0.3); 0.20% a month as a portfolio (t 1.0) and 0.04% (t 0.2) | Not established |

Why the routine split probably differs:

- Short history. The filings start in mid-2003, so labels start in 2006. The paper has 19 years; the early period here has 77 months.
- Small portfolios. The opportunistic and routine buy portfolios hold about 21 and 20 stocks a month here.
- Survivors only. The firms that failed are missing from both portfolios.
- Different period. Most of this sample is after the paper circulated in 2010.
- The post-2012 CAPM alphas are negative for nearly everything, including the benchmark of all liquid stocks (-0.31% a month). That is small stocks trailing SPY, and says little about insiders.

I cannot tell these apart with this data.

## All horizons

Mean abnormal return for all clusters, with overlap-robust t-statistics. From `results/event_study.csv`.

| Benchmark | Days | Before July 2012 | From July 2012 | Change |
|---|---|---|---|---|
| Matched controls | 21 | 2.0% (t 3.0) | -0.1% (t -0.3) | -2.1 (t -2.7) |
| Matched controls | 63 | 2.5% (t 2.0) | 0.2% (t 0.3) | -2.4 (t -1.7) |
| Matched controls | 126 | 2.8% (t 2.0) | 0.1% (t 0.1) | -2.7 (t -1.6) |
| All liquid stocks | 21 | 2.1% (t 3.4) | 0.1% (t 0.1) | -2.0 (t -2.8) |
| All liquid stocks | 63 | 2.8% (t 2.4) | 0.4% (t 0.7) | -2.4 (t -1.8) |
| All liquid stocks | 126 | 2.8% (t 2.4) | 0.3% (t 0.3) | -2.6 (t -1.7) |
| SPY | 63 | 5.0% (t 2.2) | 0.7% (t 0.7) | -4.3 (t -1.8) |
| IWM | 63 | 4.1% (t 2.1) | 1.0% (t 1.8) | -3.1 (t -1.5) |

The 21-day numbers are the strongest in the table. Most of the early mean arrives in the first month, and the 21-day change has t = -2.7. The headline horizon was fixed at 63 days in the first run, so the 21-day row is one of twelve cells and I do not treat it as the result. It is still what a reader should look at first if better data becomes available.

## What was chosen when

Git history shows the order of commits. It cannot show what I had seen when I chose something, so this section says only what the commits support.

In the first run (commits f9892f4 and 5a346bf, 23 November 2026): split date 1 July 2012, the month after the Journal of Finance issue. Three insiders, 30-day window, 182-day cooldown. Horizons 21, 63 and 126 trading days, with 63 as the headline. The all-liquid-stocks benchmark as the headline, with SPY and IWM beside it. Price floor $2, dollar-volume floor $100,000. Cost 1% per round trip. Events from 1 July 2003 to 30 September 2025, prices to 30 June 2026. Month-clustered standard errors. Four other cluster definitions, all run in the same pass. All of these are in the code of f9892f4 and none has changed since.

Added after those results were public, in answer to a review, and so post hoc:

- The quarter-clustered and overlap-robust standard errors, and the lag rule.
- Matched controls: five controls, dollar-volume matching, the 182-day exclusion.
- The block bootstrap for medians: six-month blocks, 2,000 draws.
- The rows without 2009, the winsorised rows at 1% and 5%, and the 2007 to 2014 window. I picked 2009 because the first results showed its mean of 11.1% beside a median of -0.9%. I picked 2007 to 2014 from the coverage table: the share of clusters priced is between 19% and 26% in each of those years, on both sides of the split date.
- The survivorship bounds and their grid of losses and shares.
- The split-aware price check, its 30-day window and 10% tolerance, and the flag on each event.
- Leading with matched controls and the monthly portfolio in the README. With the first results in hand, the choice of which number to lead with is itself post hoc. Every benchmark is reported side by side so the reader can pick another.

Nothing in the first specification was changed. The unreconciled events stay in the main sample for the same reason.

One thing changed during the first build. The first full run failed on ticker catalogue rows with no CIK, and the fix was to skip those rows. After the first results I added one row to the calendar-time table, the benchmark of all liquid stocks, to show its own CAPM alpha.

## Variants

Five cluster definitions, abnormal return over 63 days. From `results/variants.csv`.

| Insiders | Window | Benchmark | Before July 2012 | From July 2012 |
|---|---|---|---|---|
| 3 | 30 days | All liquid stocks | 2.8% (t 2.4, n 1,706) | 0.4% (t 0.7, n 3,205) |
| 3 | 30 days | Matched controls | 2.5% (t 2.0) | 0.2% (t 0.3) |
| 2 | 30 days | All liquid stocks | 1.3% (t 3.5, n 3,220) | 0.3% (t 0.7, n 6,125) |
| 2 | 30 days | Matched controls | 0.9% (t 1.6) | 0.2% (t 0.3) |
| 4 | 30 days | All liquid stocks | 3.1% (t 2.3, n 1,016) | 0.5% (t 0.8, n 1,886) |
| 4 | 30 days | Matched controls | 2.9% (t 1.9) | 0.2% (t 0.4) |
| 3 | 90 days | All liquid stocks | 1.7% (t 2.0, n 2,113) | 0.1% (t 0.1, n 3,875) |
| 3 | 90 days | Matched controls | 1.2% (t 1.5) | -0.1% (t -0.2) |
| 3 | 10 days | All liquid stocks | 2.8% (t 2.4, n 1,358) | 0.4% (t 0.7, n 2,668) |
| 3 | 10 days | Matched controls | 2.8% (t 2.2) | -0.1% (t -0.2) |

## Differences from the first design

- The first design named the SEC's Insider Transactions Data Sets. The run uses the Layline mirror of the same filings, because the SEC asks for a contact identity with each request. A reader for the SEC files existed until this revision. It had only been run on hand-built files and its column names came from my recollection of the SEC's description, so it has been removed.
- "Before and after 2012" is implemented as before and after 1 July 2012.
- "Against the market" became four benchmarks.

## Not verified

- The Lakonishok and Lee tables. Only the working paper abstract was read.
- The journal versions of both papers. Numbers quoted from Cohen, Malloy and Pomorski are from the 2010 working paper.
- The Barber and Lyon and Newey and West formulas against their papers. Both are standard and are tested against hand calculations.
- The finite-sample behaviour of the overlap-robust standard error beyond the one simulation in the tests. With about 107 months before July 2012 it may still be somewhat too small.
- How complete the Layline tables are against EDGAR itself. The dataset ships a short errors file that is not used here.
- How complete the split table is. A missing split shows up as an unreconciled event, and I have not separated those from wrong joins.
- Whether a later price snapshot gives similar numbers.
- Whether the results are byte-identical on another machine.
- Whether the price provider's terms allow the Hugging Face dataset to republish Yahoo Finance data. The dataset is published under ODC-BY by its maintainer; nothing from it is committed here.
- That no free source of delisted prices exists. I searched the web and the Hugging Face dataset index and found only paid or keyed sources.
