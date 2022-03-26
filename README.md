# Detection of Ransomware Attack Families via Bitcoin Transactions

Authors : Rihem Mansri - Mohamed Issa - Mootez Dakhlaoui - Bourhan Dernayka - Joel Pascal Soffo - Ronny Tonato

Over the past four years, the ransom demanded by hackers increased by a shocking 2,966.66 percent. In 2021, the average ransom demand reached $ 220,298 — up 43 percent compared to 2020. The explosive growth in ransomware demand was in 2019, where the average ransom demand grew 14 times, up from 6,000 in 2018 to 84,000 by the end of the year. In this challenge, the goal will be to create a model that is able to detect whether the given addresses belong to any of the attacker families by tracing the cryptocurrency transactions in the entire Bitcoin transaction graph from 2009 to 2018.

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started with the [dedicated notebook](bitcoin_heist_starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
