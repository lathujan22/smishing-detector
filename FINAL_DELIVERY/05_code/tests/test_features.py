"""Step 5 unit tests - 10 hand-made messages.   Run:  python -m unittest -v tests.test_features"""
import unittest

from src.features import FLAG_NAMES, extract_features, flag_vector
from src.preprocess import preprocess


class TestPreprocessAndFeatures(unittest.TestCase):

    def test_01_english_bank_smishing(self):
        f = extract_features("URGENT! Your BOC account has been suspended. Verify within 24 hours: http://boc-verify.top/login")
        self.assertEqual((f["has_url"], f["risky_domain"], f["action_request"]), (1, 1, 1))
        self.assertGreaterEqual(f["urgency"], 0.67)
        self.assertEqual(f["impersonation"], 1.0)
        self.assertGreaterEqual(f["link_risk"], 0.9)

    def test_02_english_otp_ham(self):
        f = extract_features("Your BOC OTP is 482913. Do not share this code with anyone.")
        self.assertEqual((f["has_url"], f["action_request"], f["urgency"]), (0, 0, 0))
        self.assertLessEqual(f["impersonation"], 0.4)          # brand named, nothing asked

    def test_03_genuine_promo_official_link(self):
        f = extract_features("Get 5GB FREE! Recharge Rs.500 now: https://reloadpay.hutch.lk/en T&C apply.")
        self.assertEqual((f["has_url"], f["risky_domain"]), (1, 0))
        self.assertLessEqual(f["link_risk"], 0.1)
        self.assertLessEqual(f["impersonation"], 0.3)

    def test_04_sinhala_parcel_smishing(self):
        msg = "ශ්‍රී ලංකා තැපෑල: ඔබගේ පාර්සලය රඳවා ඇත. වහාම රු. 150 ගෙවන්න: slpost-lk.top"
        p, f = preprocess(msg), extract_features(msg)
        self.assertEqual(p["language"], "si")
        self.assertIn("්‍", p["text_clean"])        # ශ්‍රී keeps its joiner
        self.assertEqual((f["risky_domain"], f["has_money"], f["action_request"]), (1, 1, 1))
        self.assertGreater(f["urgency"], 0)
        self.assertGreaterEqual(f["impersonation"], 0.7)

    def test_05_tamil_otp_ham(self):
        msg = "உங்கள் OTP எண் 482914. இதை யாருடனும் பகிர வேண்டாம்."
        self.assertEqual(preprocess(msg)["language"], "ta")
        self.assertEqual(extract_features(msg)["action_request"], 0)

    def test_06_singlish_loan_lure_with_operator_warning(self):
        msg = "DIALOG WARNING - SPAM : Card ekata ikman naya. Rs. 300 000 dakwa labaganna https://f1n8.com/wk0hSI2"
        p, f = preprocess(msg), extract_features(msg)
        self.assertEqual(p["telco_flag"], 1)
        self.assertNotIn("warning", p["text_model"])            # label leak removed
        self.assertEqual(p["language"], "si-en")
        self.assertEqual((f["has_url"], f["has_money"], f["action_request"]), (1, 1, 1))

    def test_07_tanglish_customs_smishing(self):
        msg = "Unga parcel customs-la maatikittu irukku. Udane Rs.1500 kattunga: bit.ly/abc12"
        p, f = preprocess(msg), extract_features(msg)
        self.assertEqual(p["language"], "ta-en")
        self.assertEqual((f["url_shortener"], f["action_request"]), (1, 1))
        self.assertGreater(f["urgency"], 0)

    def test_08_personal_ham_all_zero(self):
        self.assertEqual(flag_vector("Hey, are we still meeting at 5? Bring the notes."), [0.0] * len(FLAG_NAMES))

    def test_09_masking(self):
        p = preprocess("Call 0771234567 or visit www.example.com, email a@b.com 🎉 Rs.500 only")
        for tok in ["<PHONE>", "<URL>", "<EMAIL>", "<EMOJI>", "<MONEY>"]:
            self.assertIn(tok, p["text_model"])
        self.assertNotRegex(p["text_model"], r"\d{4,}")        # no personal numbers left
        self.assertEqual(p["urls"], ["www.example.com"])         # 'Rs.500' is not a URL

    def test_10_fake_government_lookalike(self):
        f = extract_features("[Traffic Police] Pay your fine within 24 hours or face legal action: https://lktel-com.cc/lk")
        self.assertEqual((f["risky_domain"], f["action_request"]), (1, 1))
        self.assertGreaterEqual(f["urgency"], 0.67)
        self.assertGreaterEqual(f["impersonation"], 0.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
