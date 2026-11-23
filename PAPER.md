# Sources, formulas and comparison

## Citations

1. Lakonishok, J. and Lee, I. (2001). "Are Insider Trades Informative?" Review of Financial Studies 14(1), 79-111. I could not reach the journal text. I read the abstract of the working paper version (NBER Working Paper 6656, July 1998), which is a scanned PDF, and nothing else from it. No formula here is taken from this paper.
2. Cohen, L., Malloy, C. and Pomorski, L. (2012). "Decoding Inside Information." Journal of Finance 67(3), 1009-1043. I read the working paper version (NBER Working Paper 16454, October 2010). Page and table numbers below refer to that version and may differ from the journal.
3. Balogh, A. (2023). "Insider trading." Scientific Data 10. doi:10.1038/s41597-023-02147-6. The dataset is doi:10.7910/DVN/VH6GVH. The article page would not load without a sign-in redirect, so the table layout was read from the files themselves.
4. Barber, B. and Lyon, J. (1997). "Detecting long-run abnormal stock returns: The empirical power and specification of test statistics." Journal of Financial Economics 43(3), 341-372. Not fetched in this session. The buy-and-hold abnormal return is written from memory of its equation 3.
5. Newey, W. and West, K. (1987). "A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica 55(3), 703-708. Not fetched in this session. The estimator is the standard Bartlett-kernel form and is checked against a hand calculation in the tests.
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

Three benchmarks `b` are used: SPY, IWM, and an equal-weight buy-and-hold portfolio of every liquid stock in the price file formed at the same close. The last one was chosen as the headline before the first run, because it shares the survivorship of the event stocks.

Standard errors for the mean are clustered by the calendar month of entry:

```
se = sqrt( sum_g ( sum_{i in g} (x_i - mean) )^2 ) / n
```

The difference between periods uses `sqrt(se_pre^2 + se_post^2)`. Returns over 63 and 126 days overlap across neighbouring months, which this does not correct. The calendar-time portfolios below do.

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

The paper also reports Fama-French, Carhart, DGTW and five-factor alphas and value-weighted portfolios. None of those are built here: there are no factor returns or market capitalisations in the data used.

## Claim by claim

| Claim in the source | Source number | Here | Match |
|---|---|---|---|
| Opportunistic buys minus routine buys, equal weight, average monthly return | 0.68% (1989-2007), Table IV panel A | 0.16% before July 2012 (t 0.5, 77 months); -0.08% after (t -0.3, 160 months); 0.00% overall | No |
| Opportunistic buys, equal weight, CAPM alpha | 1.51% a month (t 5.89) | 0.61% before (t 1.5); -0.55% after (t -1.8) | No |
| Routine buys, equal weight, CAPM alpha | 0.92% a month (t 4.34) | 0.48% before (t 1.4); -0.38% after (t -1.5) | No |
| Opportunistic buys minus opportunistic sells, equal weight | 1.55% a month average | 0.49% before (t 1.2); -0.46% after (t -2.0) | No |
| The classified sample is about one third of all insider transactions | section I | 15% of purchases before July 2012 and 21% after | Lower |
| Purchases are informative (Lakonishok and Lee abstract) | qualitative | Firms with any officer or director purchase beat the matched benchmark by 0.19% a month before July 2012 (t 1.6) and 0.07% after (t 0.6) | Weak |
| Clusters of buyers predict returns | no source number | 2.8% over 63 days before July 2012 (t 2.7); 0.4% after (t 0.7) | Before only |

Why the routine split probably differs:

- Short history. The filings start in mid-2003, so labels start in 2006. The paper has 19 years; the early period here has 77 months.
- Small portfolios. The opportunistic and routine buy portfolios hold about 21 and 20 stocks a month here.
- Survivors only. The firms that failed are missing from both portfolios.
- Different period. Most of this sample is after the paper circulated in 2010.
- The post-2012 CAPM alphas are negative for nearly everything, including the benchmark of all liquid stocks (-0.31% a month). That is small stocks trailing SPY, and says little about insiders.

I cannot tell these apart with this data.

## Differences from the plan

- The design named the SEC's Insider Transactions Data Sets. The published run uses the Layline mirror of the same filings, because the SEC asks for a contact identity with each request. The direct reader is in `quorum.edgar` and has only been run on hand-built files. Its column names are from my recollection of the SEC's description of those files and need checking on a first real download.
- "Before and after 2012" is implemented as before and after 1 July 2012, the month after the Journal of Finance issue.
- "Against the market" became three benchmarks, with the matched one as the headline.

## Choices fixed before the first run

Split date 1 July 2012. Three insiders, 30-day window, 182-day cooldown. Horizons 21, 63 and 126 trading days, with 63 as the headline. Matched benchmark as the headline. Price floor $2, dollar-volume floor $100,000. Cost 1% per round trip. Events from 1 July 2003 to 30 September 2025, prices to 30 June 2026.

## Variants tried

Five cluster definitions were run, all in one pass and all reported in `results/variants.csv`. Abnormal return over 63 days against the matched benchmark:

| Insiders | Window | Before July 2012 | From July 2012 |
|---|---|---|---|
| 3 | 30 days | 2.8% (t 2.7, n 1,706) | 0.4% (t 0.7, n 3,205) |
| 2 | 30 days | 1.3% (t 3.4, n 3,220) | 0.3% (t 0.8, n 6,125) |
| 4 | 30 days | 3.1% (t 2.6, n 1,016) | 0.5% (t 0.8, n 1,886) |
| 3 | 90 days | 1.7% (t 2.2, n 2,113) | 0.1% (t 0.1, n 3,875) |
| 3 | 10 days | 2.8% (t 2.6, n 1,358) | 0.4% (t 0.7, n 2,668) |

The first row is the main specification and was set before any of them ran. Three benchmarks and three horizons were also computed for it, nine combinations, all in `results/event_study.csv`.

One thing changed during the build. The first full run failed on ticker catalogue rows with no CIK, and the fix was to skip those rows. No result existed before that fix. After the first results I added one row to the calendar-time table, the benchmark of all liquid stocks, to show its own CAPM alpha. Nothing in the specification was changed after seeing numbers.

## Not verified

- The Lakonishok and Lee tables. Only the working paper abstract was read.
- The journal versions of both papers. Numbers quoted from Cohen, Malloy and Pomorski are from the 2010 working paper.
- The Barber and Lyon and Newey and West formulas against their papers. Both are standard and are tested against hand calculations.
- The SEC quarterly file layout against a real file.
- How complete the Layline tables are against EDGAR itself. The dataset ships a short errors file that is not used here.
- The share of price mismatches in `results/mapping_check.csv` that are wrong joins.
- Whether the price provider's terms allow the Hugging Face dataset to republish Yahoo Finance data. The dataset is published under ODC-BY by its maintainer; nothing from it is committed here.
