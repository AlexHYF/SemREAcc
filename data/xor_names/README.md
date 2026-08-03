# Crossed Chinese/Japanese name benchmark

This benchmark represents the predicate

```text
(Chinese given name AND Japanese surname)
OR
(Japanese given name AND Chinese surname)
```

Every example uses the displayed order `given-name surname`, regardless of the
customary domestic name order in China or Japan. `C_J` and `J_C` rows have label
`1`; `C_C` and `J_J` rows have label `0`.

## Files

- `vocab/*.csv`: four 100-name source vocabularies with source ranks.
- `xor_names_full.csv`: all 40,000 Cartesian products, 10,000 per pair type.
- `xor_names_core.csv`: a balanced 4,000-row evaluation set, 1,000 per pair
  type. Ten deterministic rotations make every vocabulary item occur ten times
  within every applicable pair type.
- `individual_queries.csv`: 800 cached semantic atoms. Each of the 200 given
  names and 200 surnames is tested against both Chinese and Japanese predicates.

Regenerate the derived CSVs with:

```bash
python3 scripts/build_xor_names_dataset.py
```

## Evaluation protocol

For component-wise classification, query each row of `individual_queries.csv`
once per model and cache the Boolean result. A compatible prompt template is:

```text
Treat "{name}" as a romanized {role}, not as evidence about a person's
nationality. Is it commonly used as a {target_origin} {role}?
Answer only YES or NO.
```

Here `{role}` is `given name` or `surname`, and `{target_origin}` is `Chinese`
or `Japanese`. Reconstruct a pair's prediction as:

```text
(ChineseGiven(first) AND JapaneseSurname(last)) OR
(JapaneseGiven(first) AND ChineseSurname(last))
```

For direct whole-name classification, use the same definitions:

```text
The text "{full_name}" contains two romanized tokens in GIVEN-NAME SURNAME
order. Does it satisfy either of these conditions?
1. The first token is commonly used as a Chinese given name and the second as
   a Japanese surname.
2. The first token is commonly used as a Japanese given name and the second as
   a Chinese surname.
Answer only YES or NO.
```

Run the 4,000-row core set first. The exhaustive set is intended for a later
stress test because direct classification requires one model invocation for
every row, whereas the component method needs only 800 unique cached calls.

## Sources

- Chinese given names: [Forebears, Most Popular First Names in
  China](https://forebears.io/china/forenames). The list uses 100 entries from
  source ranks 2--102. Rank 1 (`Nushi`) and rank 60 (`Laoshi`) were excluded as
  likely extraction artifacts/honorifics rather than personal given names.
- Chinese surnames: [Forebears, Most Common Last Names in
  China](https://forebears.io/china/surnames), ranks 1--100.
- Japanese given names: [Forebears, Most Popular First Names in
  Japan](https://forebears.io/japan/forenames), ranks 1--100. The list is also
  archived and described by Ngai, Kilpatrick, and Ćwiek's open-access study and
  [OSF dataset](https://osf.io/yrx4u/). The paper reports that the data came from
  a 2014 telephone directory and was inspected by a native Japanese linguist.
- Japanese surnames: [Forebears, Most Common Last Names in
  Japan](https://forebears.io/japan/surnames), ranks 1--100. A separate
  Meiji-Yasuda survey also reports that Japan's top 100 surnames cover slightly
  more than one third of the population.

## Scope and limitations

The labels mean membership in these romanized source lists, not a claim about a
person's nationality or ethnicity. Romanization loses the original characters
and tones/readings, so a surface string may be valid in more than one culture or
may correspond to several native-script names. The selected Chinese and
Japanese vocabularies have no exact case-insensitive overlap within the same
name role, yielding a deliberately clean first benchmark. Later experiments
should add ambiguous and cross-listed names as a separate hard subset.

The source frequency data is observational and not demographically balanced.
It is appropriate for testing name-origin prompts, but not for inferring the
identity of real people.
