from django.test import SimpleTestCase

from ml.classification.predict import classify
from ml.similarity.duplicates import find_similar


class ClassifierTests(SimpleTestCase):
    def test_obvious_cases(self):
        cases = {
            "There is no water supply in our lane for five days": "Water Supply",
            "My pension has not been credited for three months": "Pension",
            "Transformer is sparking and there is a power cut": "Electricity",
            "Garbage is not collected and the drain is overflowing": "Sanitation & Garbage",
        }
        for text, expected in cases.items():
            self.assertEqual(classify(text).category, expected, text)

    def test_confidence_and_top3(self):
        r = classify("ration shop is not giving wheat")
        self.assertTrue(0 <= r.confidence <= 1)
        self.assertEqual(len(r.top3), 3)
        self.assertAlmostEqual(r.top3[0][1], r.confidence)

    def test_empty_text_is_unclassified(self):
        self.assertEqual(classify("!!! 123").category, "Unclassified")


class SimilarityTests(SimpleTestCase):
    corpus = [(1, "No water supply in Sector 15 for three days", "Sector 15"),
              (2, "Pension not credited for two months", "Ward 4"),
              (3, "No water supply in Sector 15 for a week", "Gandhi Nagar")]

    def test_same_place_is_duplicate(self):
        m = find_similar("There is no water supply in Sector 15 since three days", self.corpus, location="Sector 15")
        self.assertEqual(m[0].id, 1)
        self.assertEqual(m[0].kind, "duplicate")

    def test_other_place_is_only_related(self):
        m = {x.id: x for x in find_similar("No water supply in Sector 15 for a week", self.corpus, location="Sector 15")}
        self.assertEqual(m[3].kind, "related")

    def test_unrelated_not_returned(self):
        ids = [x.id for x in find_similar("Pension not credited for two months", self.corpus)]
        self.assertNotIn(1, ids)

    def test_empty_corpus(self):
        self.assertEqual(find_similar("anything at all here", []), [])
