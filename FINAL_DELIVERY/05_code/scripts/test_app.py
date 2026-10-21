"""End-to-end test of the running web app (python scripts/test_app.py). Uses only the standard library."""
import json
import urllib.request

U = "http://127.0.0.1:8000"


def call(path, body=None):
    req = urllib.request.Request(U + path, data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


print("health:", call("/health"))
S = call("/samples")["samples"]
R = call("/predict/batch", {"texts": [s["text"] for s in S]})["results"]
ok_s = ok_x = 0; lang_ok = 0
for s, r in zip(S, R):
    x = r["xlmr"]["prediction"] if r["xlmr"] else "-"
    ok_s += r["svm"]["prediction"] == s["expected"]; ok_x += x == s["expected"]
    lang_ok += r["language"]["code"] == s["language"] or (s["language"] == "ta-en" and r["language"]["code"] in ("ta-en", "en"))
    print(f'{s["id"]} {s["language"]:6} expected={s["expected"]:11} svm={r["svm"]["prediction"]:11} xlmr={x:11} lang={r["language"]["code"]}')
print(f"LinearSVC {ok_s}/{len(S)} | XLM-R {ok_x}/{len(S)} | language detected as written {lang_ok}/{len(S)}")
m = call("/metrics"); print("metrics:", len(m["comparison"]), "rows,", len(m["figures"]), "figures")
print("model/info xlmr:", call("/model/info")["xlmr_status"])
