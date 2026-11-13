"""
ml/preprocessing/synthetic_corpus.py
==============================================
Generates an ILLUSTRATIVE, SYNTHETIC complaint corpus (14 categories) so the
whole system runs end-to-end when the Kaggle file is not present.

*** These complaints are invented. They are NOT CPGRAMS records and they are NOT
    the Kaggle dataset. ***  Every row carries source="synthetic".

Design (so that evaluation is not trivially 100 %):
  * each category has several complaint "templates" with slot fillers,
  * a share of sentences borrow generic phrases used by every category,
  * a few categories deliberately share vocabulary (Water vs Sanitation,
    Pension vs Banking, Electricity vs Roads) so there is real confusion,
  * light typos / casing noise is injected,
  * "duplicate clusters": 1 original + 1-3 paraphrases of the SAME incident get
    the same cluster_id, giving ground truth for duplicate detection.

If you place the real Kaggle CSV in ml/data/raw/ the loader in
load_complaints.py uses it instead and this file is not used.
"""

from __future__ import annotations

import random

AREAS = ["Sector 15", "Gandhi Nagar", "Model Town", "Civil Lines", "Rajiv Colony", "Shastri Park",
         "Lajpat Nagar", "Ward 12", "Block C", "Nehru Enclave", "Indira Vihar", "Kalyan Puri",
         "MG Road", "Ashok Vihar", "Subhash Chowk"]
DURATIONS = ["two days", "three days", "a week", "ten days", "two weeks", "almost a month", "since Monday"]
OPEN = ["I want to report that", "Kindly look into the issue that", "This is to complain that",
        "Respected officer,", "I am writing to inform that", "Urgent attention needed:"]
CLOSE = ["Please take action.", "Request immediate resolution.", "Nobody has responded to us.",
         "Many residents are facing this.", "We have complained earlier also.", "Kindly resolve at the earliest."]

# category -> list of templates ({a}=area, {d}=duration)
T = {
 "Water Supply": [
  "there is no water supply in {a} for {d}", "the tap water in {a} is dirty and smells bad since {d}",
  "water pressure is extremely low in our lane in {a}", "the main pipeline is leaking near {a} and water is wasted",
  "tanker water has not arrived in {a} for {d}", "the water supply timing is irregular in {a} and the pipe is broken"],
 "Sanitation & Garbage": [
  "garbage is not being collected in {a} for {d}", "the drain is blocked and sewage is overflowing in {a}",
  "dustbins near {a} market are overflowing and smelling", "open drain near {a} school is causing mosquitoes",
  "the sweeper does not clean the street in {a}", "dirty water from the sewer line is on the road in {a}"],
 "Roads & Infrastructure": [
  "there are big potholes on the road in {a} for {d}", "the street light is not working on the road at {a}",
  "the bridge near {a} has cracks and is unsafe", "road repair work in {a} is left incomplete",
  "footpath in {a} is broken and people are falling", "no speed breaker near the school at {a}"],
 "Electricity": [
  "there is frequent power cut in {a} for {d}", "the electricity transformer in {a} is sparking",
  "wrong electricity bill received for this month in {a}", "a live wire is hanging from the pole at {a}",
  "voltage is very low in {a} and appliances are damaged", "new electricity meter connection pending for {d} in {a}"],
 "Pension": [
  "my pension has not been credited for {d}", "old age pension application is pending since {d}",
  "family pension not released after the death of my husband", "pension amount is less than the sanctioned amount",
  "life certificate was submitted but the pension stopped", "pension arrears are not paid for {d}"],
 "Taxation": [
  "my income tax refund is pending for {d}", "wrong tax demand notice received for the assessment year",
  "TDS not reflected in my form 26AS", "property tax bill has an incorrect amount for {a}",
  "GST registration application is stuck for {d}", "tax assessment order has an error and needs correction"],
 "Documentation": [
  "my caste certificate application is pending for {d}", "the birth certificate has a spelling mistake in name",
  "domicile certificate not issued even after {d}", "passport verification is delayed for {d}",
  "ration card correction request is not processed", "aadhaar address update not done even after {d}"],
 "Public Distribution": [
  "the ration shop in {a} is not giving full wheat quota", "ration dealer demands extra money for ration",
  "fair price shop is closed most days in {a}", "my name is missing from the ration list",
  "rice supplied at the ration shop is of poor quality", "LPG cylinder not delivered for {d}"],
 "Healthcare": [
  "the government hospital in {a} has no doctor available", "medicines are not available at the dispensary in {a}",
  "ambulance did not arrive for the emergency call", "the hospital staff refused treatment to my father",
  "Ayushman card is not accepted by the hospital", "primary health centre in {a} remains closed"],
 "Education": [
  "the school in {a} has no teachers for {d}", "scholarship amount is not received for this year",
  "college has not issued my marksheet for {d}", "university result is wrongly declared and needs recheck",
  "mid day meal quality is poor in the school at {a}", "admission was denied unfairly in the government school"],
 "Police & Public Safety": [
  "police did not register my FIR for the theft", "there is regular harassment of women near {a}",
  "illegal liquor is sold openly in {a} and police is silent", "no police patrolling in {a} at night",
  "my complaint about a threat is not investigated", "traffic police is demanding bribe at {a} crossing"],
 "Railways": [
  "train {n} runs late every day and passengers suffer", "railway refund for the cancelled ticket is not received",
  "toilets in the train coach were very dirty", "the waiting room at the station of {a} is locked",
  "ticket counter staff behaved rudely with passengers", "reserved seat was occupied by others in train {n}"],
 "Telecom & Postal": [
  "mobile network is very weak in {a} for {d}", "the speed post parcel is not delivered for {d}",
  "broadband connection is not working for {d}", "post office in {a} does not give money order payment",
  "telephone landline is dead in {a} since {d}", "postman does not deliver letters to our address"],
 "Banking & Finance": [
  "bank has not credited the subsidy amount for {d}", "loan application is rejected without any reason",
  "ATM in {a} remains out of cash most days", "wrong amount was debited from my account",
  "bank is not opening a jan dhan account for me", "KCC loan sanction is delayed for {d}"],
}
GENERIC_SHARED = {   # sentences that confuse category boundaries a little
 "Water Supply": ["The sewage smell is also coming with the tap water."],
 "Sanitation & Garbage": ["There is also a water pipe leaking nearby."],
 "Pension": ["The bank says the amount was not received from the department."],
 "Banking & Finance": ["The pension subsidy is also pending from the department."],
 "Electricity": ["The road is dark and unsafe because of this."],
 "Roads & Infrastructure": ["The electric pole is also leaning dangerously."],
}


def _noise(s: str, rng: random.Random) -> str:
    if rng.random() < 0.15:
        words = s.split()
        i = rng.randrange(len(words))
        w = words[i]
        if len(w) > 4:
            j = rng.randrange(1, len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        s = " ".join(words)
    if rng.random() < 0.1:
        s = s.lower()
    return s


def _render(tpl: str, rng: random.Random, area=None, dur=None) -> str:
    return tpl.format(a=area or rng.choice(AREAS), d=dur or rng.choice(DURATIONS), n=rng.randint(12001, 12999))


def generate(n_total: int = 614, seed: int = 7) -> list[dict]:
    """Return rows: {text, category, cluster_id, location, source}."""
    rng = random.Random(seed)
    cats = list(T.keys())
    rows, cluster = [], 0
    per_cat = n_total // len(cats)
    for c in cats:
        made = 0
        while made < per_cat + (1 if len(rows) < n_total - per_cat * len(cats) else 0):
            cluster += 1
            tpl = rng.choice(T[c])
            area, dur = rng.choice(AREAS), rng.choice(DURATIONS)
            size = rng.choices([1, 2, 3, 4], weights=[5, 3, 2, 1])[0]
            for k in range(size):
                tpl_k = tpl if k == 0 else rng.choice(T[c])
                body = _render(tpl_k, rng, area, dur if k == 0 else None) if k == 0 \
                    else _render(tpl, rng, area, dur)  # paraphrase = same incident, different wrapper
                parts = [rng.choice(OPEN), body + ".", rng.choice(CLOSE)]
                if c in GENERIC_SHARED and rng.random() < 0.3:
                    parts.insert(2, rng.choice(GENERIC_SHARED[c]))
                if k > 0:
                    rng.shuffle(parts[1:2])
                text = _noise(" ".join(parts), rng)
                rows.append(dict(text=text, category=c, cluster_id=cluster, location=area, source="synthetic"))
                made += 1
                if made >= per_cat + 1:
                    break
    rng.shuffle(rows)
    return rows[:n_total]


if __name__ == "__main__":
    import collections
    r = generate()
    print(len(r), collections.Counter(x["category"] for x in r).most_common(3))
    print(r[0])
