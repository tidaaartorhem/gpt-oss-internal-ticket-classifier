# Internal Process Ticket Classifier (GPT‑OSS Demonstration)

## Motivation

Large finance and insurance organizations rely on internal software
teams to build and maintain critical systems.  These teams must
handle a constant influx of tickets – bug reports, feature requests,
compliance tasks, security patches and other work items.  In popular
projects tens or hundreds of issues can be reported every day【766327437034919†L55-L57】.
Manual labeling of these tickets is labour‑intensive and error‑prone【766327437034919†L59-L70】,
yet consistent labels help developers prioritise work and improve
process efficiency【766327437034919†L27-L37】.  Automating the classification of
tickets can therefore streamline internal processes and reduce
administrative overhead.

OpenAI’s **GPT‑OSS** family provides open‑weight reasoning models
specifically designed for advanced tool use and on‑premises
deployment.  The gpt‑oss‑120b and gpt‑oss‑20b models were released
under an Apache 2.0 license with open weights, representing a
significant shift towards openness【180865669010467†L20-L33】.  The larger
gpt‑oss‑120b model uses a Mixture‑of‑Experts (MoE) architecture with
36 layers and 128 experts, activating only four experts per token so
that only about 5.1 billion of its 117 billion parameters are used
per inference step【180865669010467†L43-L50】.  Despite the enormous scale, the
model can run on a single 80 GB GPU and supports context windows up
to 128 k tokens【180865669010467†L62-L74】.  These properties make GPT‑OSS
well‑suited for enterprise use cases where long documents and
offline deployment are important.  While the full 120 b parameter
model cannot be run in this environment, this project demonstrates
how such an advanced model could be used to classify internal
tickets through instruction‑based prompting.

## Dataset

The repository includes a synthetic dataset,
`internal_process_tickets.csv`, containing 54 tickets across nine
categories relevant to software teams in regulated industries:

| Category | Description (examples) |
| --- | --- |
| **Compliance Update** | Tickets that request updates or new features required by regulations (e.g., updating data retention settings or adding audit‑trail support). |
| **Security Vulnerability** | Tasks related to patching vulnerabilities, rotating keys or performing security reviews. |
| **Feature Request** | New product features requested by internal stakeholders (e.g., dashboards, multi‑factor authentication, integration APIs). |
| **Bug** | Defects discovered in the software (null pointer exceptions, incorrect calculations, UI glitches). |
| **Performance Improvement** | Optimizations for speed, scalability or resource usage. |
| **Technical Debt** | Refactoring, removal of deprecated code and adding tests or documentation to reduce maintenance burden. |
| **Risk Assessment** | Analytical tasks to evaluate market volatility, default risk or perform stress testing. |
| **Infrastructure Upgrade** | Upgrades to infrastructure and platforms such as cloud migration or database upgrades. |
| **Documentation** | Tickets requesting or updating documentation and onboarding materials. |

Each row in the dataset contains two columns: `ticket` (free‑text
description) and `category` (the ground‑truth label).

## Methodology

Due to resource constraints in this environment, we implement a
baseline classifier using traditional machine‑learning techniques.  The
script `internal_process_classifier.py` performs the following steps:

1. **Load data** – reads the CSV file into memory.
2. **Train/test split** – splits tickets and labels into training and
   test sets using stratified sampling (75 % training, 25 % test).
3. **Vectorization** – converts ticket text into TF–IDF vectors with
   unigrams and bigrams, removing English stop words.
4. **Model training** – fits a multinomial logistic regression
   classifier to the training data.
5. **Evaluation** – evaluates the classifier on the test set and
   prints accuracy and a detailed classification report.  Predictions
   are saved to `predictions.csv`.

When run on the synthetic dataset, the baseline model achieves low
accuracy (~7 %) due to the small dataset and high variability across
classes.  This underscores the need for more sophisticated models and
a larger labelled corpus.  In practice, GPT‑OSS could be used for
few‑shot or instruction‑based classification: the model would be
prompted with a list of categories and a ticket description and asked
to output the most appropriate label.  Because GPT‑OSS models
include long‑context support and open weights, they can run locally
within enterprise environments and be fine‑tuned on confidential
internal data, addressing privacy concerns【180865669010467†L20-L33】.

## Future Work

* **Adopt GPT‑OSS for classification** – integrate the gpt‑oss‑20b or
  gpt‑oss‑120b model via the Hugging Face `transformers` library or
  other frameworks.  A few labelled examples can be embedded into the
  prompt (few‑shot learning) to enable the model to infer the correct
  ticket categories.  This approach could dramatically improve
  accuracy thanks to GPT‑OSS’s strong reasoning capabilities and
  long‑context window【180865669010467†L43-L50】.
* **Collect real data** – obtain a large, anonymised dataset of
  internal process tickets from finance or insurance institutions and
  label them using domain experts.  A more diverse dataset will allow
  training a robust classifier and evaluating GPT‑OSS against other
  models.
* **Fine‑tuning** – explore fine‑tuning GPT‑OSS on the domain
  specific dataset.  Open weights permit customisation without
  sending sensitive data to an external provider.
* **Deployment** – package the classification logic into a microservice
  that integrates with issue trackers or ticketing systems, enabling
  automatic labeling and triage.  Combined with analytics dashboards,
  this could provide insights into workload distribution and process
  bottlenecks.

## How to Run

To experiment with the baseline classifier:

```bash
# Navigate to the project directory
cd internal_process_project

# (Optional) inspect the dataset
python - <<'PY'
import pandas as pd
df = pd.read_csv('internal_process_tickets.csv')
print(df.head())
PY

# Train and evaluate the classifier
python internal_process_classifier.py --input internal_process_tickets.csv --output predictions.csv
```

If you wish to explore GPT‑OSS‑based classification, replace the
classifier in `internal_process_classifier.py` with code that calls
`transformers.AutoModelForCausalLM` and uses instruction‑based prompts.
Note that gpt‑oss‑120b requires substantial GPU memory (approx.
80 GB), while gpt‑oss‑20b can run on smaller hardware【180865669010467†L62-L74】.