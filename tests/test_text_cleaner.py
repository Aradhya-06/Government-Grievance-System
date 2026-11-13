from django.test import SimpleTestCase

from ml.preprocessing.text_cleaner import clean_text, tokenize


class CleanerTests(SimpleTestCase):
    def test_lowercase_stopwords_stemming(self):
        self.assertEqual(clean_text("Pipes are LEAKING!"), "pipe leak")

    def test_urls_numbers_removed(self):
        self.assertNotIn("http", clean_text("see https://x.com/a 12345 now"))
        self.assertNotIn("12345", clean_text("ticket 12345"))

    def test_negation_kept(self):
        self.assertIn("no", clean_text("There is no water").split())

    def test_empty_and_none(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(tokenize("1234 !!"), [])
