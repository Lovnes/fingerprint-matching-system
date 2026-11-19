"""FP-Match - fingerprint screening tool for investigations.

Run:  streamlit run app.py
"""
from __future__ import annotations

import glob
import os
import sys
import threading

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from fpmatch import __version__
from fpmatch.database import FINGERS, PrintDB, default_db_path
from fpmatch.enhance import EnhanceParams, load_image
from fpmatch.matcher import match, verdict
from fpmatch.pipeline import draw_match, draw_match_arrays, draw_minutiae, draw_template, process, sha256_bytes
from fpmatch import report

st.set_page_config(page_title="FP-Match", page_icon="🔍", layout="wide")
IMG_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_HINT = (r"C:\Users\you\Datasets\FVC2002\DB1_B" if sys.platform == "win32"
               else "/Users/you/Datasets/FVC2002/DB1_B" if sys.platform == "darwin" else "/home/you/Datasets/FVC2002/DB1_B")


def clean_folder(text: str) -> str:
    """Accept pasted paths with quotes (Windows 'Copy as path') or ~, and relative paths like samples/synthetic."""
    folder = os.path.expanduser(text.strip().strip('"\''))
    if folder and not os.path.isabs(folder) and not os.path.isdir(folder):
        folder = os.path.join(APP_DIR, folder)
    return folder


@st.cache_resource
def get_db(path):
    return PrintDB(path)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_process(data: bytes, ppi: int, latent: bool, invert: bool, crop: tuple):
    gray = load_image(data)
    roi = None
    if crop != (0, 0, 100, 100):
        h, w = gray.shape
        x0, y0, x1, y1 = crop
        roi = np.zeros((h, w), bool)
        roi[int(h * y0 / 100):int(h * y1 / 100), int(w * x0 / 100):int(w * x1 / 100)] = True
    p = process(gray, EnhanceParams(ppi=ppi, latent=latent, invert=invert), roi=roi)
    p.sha256 = sha256_bytes(data)  # hash of the original file, not the decoded pixels
    return p


# ---------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔍 FP-Match")
    st.caption(f"v{__version__} · fingerprint screening")
    page = st.radio("Section", ["Sample vs Suspect", "1:N Search", "Enroll references", "Database", "Audit log", "Help"])
    st.divider()
    st.subheader("Case")
    case = dict(case_no=st.text_input("Case / FIR no."), exhibit=st.text_input("Exhibit / lift no."),
                examiner=st.text_input("Examiner"), agency=st.text_input("Agency / lab"),
                remarks=st.text_area("Remarks", height=68))
    st.divider()
    db_path = st.text_input("Database file", value=default_db_path())
    db = get_db(db_path)
    st.caption(f"{db.count()} reference prints enrolled")
    if os.environ.get("FPMATCH_LAUNCHER"):  # started from the desktop launcher / packaged app
        st.divider()
        if st.button("⏻ Quit FP-Match", help="Stops the local server. Your database is kept."):
            st.success("FP-Match has stopped. You can close this browser tab.")
            threading.Timer(0.5, os._exit, args=(0,)).start()
            st.stop()


def print_options(key: str, default_latent=False, compact=False):
    """Per-image processing controls. Returns dict of kwargs for cached_process.

    compact=True puts every control inside one collapsed "Image settings" section."""
    if compact:
        with st.expander("⚙️ Image settings"):
            latent = st.toggle("Latent / partial print", value=default_latent, key=f"lat_{key}",
                               help="Adds background flattening, CLAHE, denoising and stricter quality filtering.")
            invert = st.toggle("Invert (light ridges)", key=f"inv_{key}",
                               help="For powder lifts on dark surfaces or negative photographs.")
            ppi = st.number_input("Resolution (ppi)", 200, 2000, 500, 50, key=f"ppi_{key}",
                                  help="Scanner / scale-photo resolution. Everything is resampled to 500 ppi.")
            st.caption("Crop to the area with ridges")
            cx = st.slider("Horizontal range (%)", 0, 100, (0, 100), key=f"cx_{key}")
            cy = st.slider("Vertical range (%)", 0, 100, (0, 100), key=f"cy_{key}")
        return dict(ppi=int(ppi), latent=latent, invert=invert, crop=(cx[0], cy[0], cx[1], cy[1]))
    c1, c2, c3 = st.columns(3)
    latent = c1.toggle("Latent / partial print", value=default_latent, key=f"lat_{key}",
                       help="Adds background flattening, CLAHE, denoising and stricter quality filtering.")
    invert = c2.toggle("Invert (light ridges)", key=f"inv_{key}",
                       help="For powder lifts on dark surfaces or negative photographs.")
    ppi = c3.number_input("Resolution (ppi)", 200, 2000, 500, 50, key=f"ppi_{key}",
                          help="Scanner / scale-photo resolution. Everything is resampled to 500 ppi.")
    crop = (0, 0, 100, 100)
    with st.expander("Crop to region of interest"):
        cx = st.slider("Horizontal range (%)", 0, 100, (0, 100), key=f"cx_{key}")
        cy = st.slider("Vertical range (%)", 0, 100, (0, 100), key=f"cy_{key}")
        crop = (cx[0], cy[0], cx[1], cy[1])
    return dict(ppi=int(ppi), latent=latent, invert=invert, crop=crop)


# Inside st.container (the bordered slots) Streamlit sizes width="stretch" images wrongly (they shrink to a
# sliver), so the slots use "content": the image's own 500 ppi size, capped at the slot width.
def show_processed(p, title, show_skel=False, width="stretch"):
    st.image(draw_minutiae(p, show_skeleton=show_skel), channels="BGR", caption=title, width=width)
    q = p.quality
    st.caption(f"Minutiae: **{p.n}** · Quality: **{q}**/100 · Ridge period: {1 / p.enh.frequency:.1f}px")
    if p.n < 12:
        st.warning("Fewer than 12 minutiae found - results will be unreliable. Try latent mode, crop or invert.")
    for n in p.enh.notes:
        st.caption("ℹ️ " + n)


def show_stages(p, width="stretch"):
    with st.expander("Processing stages"):
        c = st.columns(4) if width == "stretch" else [*st.columns(2), *st.columns(2)]  # 2x2 inside a slot
        c[0].image(p.enh.gray, caption="Input (500 ppi)", width=width)
        c[1].image((p.enh.mask * 255).astype(np.uint8), caption="Region of interest", width=width)
        c[2].image(p.enh.enhanced, caption="Gabor enhanced", width=width)
        c[3].image((~p.skeleton * 255).astype(np.uint8), caption="Thinned ridges", width=width)


def verdict_box(res):
    label, colour = verdict(res)
    fn = {"green": st.success, "orange": st.warning, "gray": st.info}[colour]
    fn(f"**{label}** — score {res.score:.1f}/100 · {res.n_supported} supported of {res.n_paired} paired minutiae")


def settings_dict(opts):
    return {"Latent mode": opts["latent"], "Inverted": opts["invert"], "Input resolution (ppi)": opts["ppi"],
            "Crop (x0,y0,x1,y1 %)": opts["crop"], "Pairing tolerance": "12 px, 20°",
            "Software": f"FP-Match v{__version__}"}


# ---------------------------------------------------------------------------- 1:1
if page == "Sample vs Suspect":
    st.header("Sample vs suspect fingerprint")
    st.write("Put the fingerprint from the scene or exhibit on the left and the suspect's fingerprint on the right.")
    show_skel = st.checkbox("Show ridge skeleton on overlays")
    left, right = st.columns(2, gap="large")
    pq = pr = None
    with left, st.container(border=True):
        st.subheader("🔎 Sample fingerprint")
        st.caption("Scene / exhibit print (latent lift, photo or scan)")
        fq = st.file_uploader("Upload sample fingerprint", type=IMG_TYPES, key="q11")
        oq = print_options("q11", default_latent=True, compact=True)
        if fq:
            with st.spinner("Processing sample…"):
                pq = cached_process(fq.getvalue(), **oq)
            show_processed(pq, f"Sample · {fq.name}", show_skel, "content"); show_stages(pq, "content")
        else:
            st.info("Upload the sample fingerprint here.")
    with right, st.container(border=True):
        st.subheader("👤 Suspect fingerprint")
        st.caption("Known print taken from the suspect (ten-print card or scan)")
        suspect = st.text_input("Suspect name / ID (optional)", key="suspect11")
        fr = st.file_uploader("Upload suspect fingerprint", type=IMG_TYPES, key="r11")
        orf = print_options("r11", compact=True)
        if fr:
            with st.spinner("Processing suspect print…"):
                pr = cached_process(fr.getvalue(), **orf)
            show_processed(pr, f"Suspect · {fr.name}", show_skel, "content"); show_stages(pr, "content")
        else:
            st.info("Upload the suspect's fingerprint here.")
    if pq and pr:
        res = match(pq.template, pr.template, pq.local, pr.local)
        st.subheader("Result")
        verdict_box(res)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score", f"{res.score:.1f}")
        m2.metric("Supported pairs", res.n_supported)
        m3.metric("Paired minutiae", res.n_paired)
        m4.metric("Rotation", f"{res.rotation_deg:.0f}°")
        canvas = draw_match(pq, pr, res)
        st.image(canvas, channels="BGR", width="stretch",
                 caption="Left: sample · Right: suspect · Green = supported pair, yellow = positional pair only")
        db.log("compare_1to1", case["examiner"], case_no=case["case_no"], query=fq.name, query_sha256=pq.sha256,
               reference=fr.name, reference_sha256=pr.sha256, suspect=suspect, score=round(res.score, 1),
               supported=res.n_supported, paired=res.n_paired)
        label, colour = verdict(res)
        settings = {f"Sample - {k}": v for k, v in settings_dict(oq).items() if k != "Software"}
        settings.update({f"Suspect - {k}": v for k, v in settings_dict(orf).items() if k not in ("Software", "Pairing tolerance")})
        settings["Software"] = f"FP-Match v{__version__}"
        html = report.build(case, "1:1 comparison - sample vs suspect",
                            dict(filename=fq.name, sha256=pq.sha256, minutiae=pq.n, quality=pq.quality,
                                 image=draw_minutiae(pq)),
                            settings,
                            [dict(label=f"{suspect} · {fr.name}" if suspect else fr.name, sha256=pr.sha256,
                                  score=res.score, paired=res.n_paired, supported=res.n_supported, verdict=label,
                                  colour=colour, image=canvas)],
                            query_title="Sample fingerprint", ref_title="Suspect fingerprint")
        st.download_button("📄 Download report (HTML)", html, file_name=f"FPMatch_{case['case_no'] or 'case'}_sample_vs_suspect.html",
                           mime="text/html", type="primary")
    else:
        st.info("The result appears here once both the sample and the suspect fingerprint are uploaded.")

# ---------------------------------------------------------------------------- 1:N
elif page == "1:N Search":
    st.header("1:N database search")
    if db.count() == 0:
        st.warning("The database is empty. Enroll reference prints first.")
    fq = st.file_uploader("Questioned print", type=IMG_TYPES, key="q1n")
    oq = print_options("q1n", default_latent=True)
    c1, c2 = st.columns(2)
    finger = c1.selectbox("Restrict to finger position", FINGERS, help="'Unknown' searches all fingers.")
    top_k = c2.slider("Candidates to show", 1, 50, 10)
    if fq:
        with st.spinner("Processing questioned print…"):
            pq = cached_process(fq.getvalue(), **oq)
        a, b = st.columns([1, 2])
        with a:
            show_processed(pq, fq.name); show_stages(pq)
        with b:
            if st.button("🔎 Search database", type="primary", disabled=db.count() == 0):
                bar = st.progress(0.0, "Searching…")
                hits = db.search(pq.template, top_k=top_k, finger=finger, progress=lambda f: bar.progress(min(f, 1.0)))
                bar.empty()
                st.session_state["hits"] = (pq.sha256, hits)
                db.log("search_1toN", case["examiner"], case_no=case["case_no"], query=fq.name,
                       query_sha256=pq.sha256, searched=db.count(), finger=finger,
                       top=[(h["subject_id"], h["print_id"], round(h["result"].score, 1)) for h in hits[:5]])
            if "hits" in st.session_state and st.session_state["hits"][0] == pq.sha256:
                hits = st.session_state["hits"][1]
                rows = []
                for k, h in enumerate(hits, 1):
                    lab, col = verdict(h["result"])
                    rows.append({"Rank": k, "Subject": h["subject_id"], "Name": h["name"], "Finger": h["finger"],
                                 "Print ID": h["print_id"], "Score": round(h["result"].score, 1),
                                 "Supported": h["result"].n_supported, "Paired": h["result"].n_paired,
                                 "Indication": lab})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
                if hits and verdict(hits[0]["result"])[1] == "gray":
                    st.info("No candidate reached the 'possible' band. The source may not be in the database, "
                            "or the questioned print is too limited.")
                pick = st.selectbox("Inspect candidate", [f"#{r['Rank']} · {r['Subject']} · {r['Finger']}"
                                                          for r in rows]) if rows else None
                report_rows = []
                for k, h in enumerate(hits, 1):
                    ref = db.get(h["print_id"])
                    rgray = db.thumbnail(h["print_id"])
                    canvas = draw_match_arrays(pq.enh.gray, pq.template, rgray, ref["template"], h["result"])
                    lab, col = verdict(h["result"])
                    report_rows.append(dict(label=f"{h['subject_id']} {h['name'] or ''} · {h['finger']} · print {h['print_id']}",
                                            sha256=ref["sha256"], score=h["result"].score, paired=h["result"].n_paired,
                                            supported=h["result"].n_supported, verdict=lab, colour=col,
                                            image=canvas if k <= 5 else None))
                    if pick and pick.startswith(f"#{k} "):
                        verdict_box(h["result"])
                        st.image(canvas, channels="BGR", width="stretch",
                                 caption="Left: questioned · Right: reference")
                html = report.build(case, f"1:N search against {db.count()} references (finger: {finger})",
                                    dict(filename=fq.name, sha256=pq.sha256, minutiae=pq.n, quality=pq.quality,
                                         image=draw_minutiae(pq)), settings_dict(oq), report_rows)
                st.download_button("📄 Download candidate report (HTML)", html,
                                   file_name=f"FPMatch_{case['case_no'] or 'case'}_search.html", mime="text/html",
                                   type="primary")

# ---------------------------------------------------------------------------- enroll
elif page == "Enroll references":
    st.header("Enroll reference prints")
    st.write("Add known prints (ten-print cards, suspect/elimination prints) to the local database.")
    src = st.radio("Source", ["Upload files", "Folder on this computer"], horizontal=True)
    o = print_options("enr")
    c1, c2 = st.columns(2)
    id_mode = c1.radio("Subject ID", ["From file name (text before first '_')", "Same ID for all"],
                       help="e.g. 101_1.tif and 101_2.tif both become subject 101.")
    fixed_id = c2.text_input("Subject ID", disabled=id_mode != "Same ID for all")
    name = c2.text_input("Name (optional)")
    finger = c1.selectbox("Finger position", FINGERS)
    notes = st.text_input("Notes (optional)")

    items = []
    if src == "Upload files":
        files = st.file_uploader("Reference prints", type=IMG_TYPES, accept_multiple_files=True)
        items = [(f.name, f.getvalue()) for f in files or []]
    else:
        folder = clean_folder(st.text_input("Folder path", placeholder=FOLDER_HINT,
                                            help="Paste the full folder path. Try samples/synthetic for the demo prints."))
        if folder and os.path.isdir(folder):
            paths = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                           if p.lower().rsplit(".", 1)[-1] in IMG_TYPES)
            st.caption(f"{len(paths)} images found")
            items = [(os.path.basename(p), p) for p in paths]
        elif folder:
            st.error("Folder not found.")
    if items and st.button(f"Enroll {len(items)} print(s)", type="primary"):
        bar = st.progress(0.0)
        added, skipped, weak = 0, 0, []
        for k, (fname, data) in enumerate(items):
            if isinstance(data, str):
                with open(data, "rb") as fh:
                    data = fh.read()
            sid = fixed_id if id_mode == "Same ID for all" else os.path.splitext(fname)[0].split("_")[0]
            if not sid:
                st.error("Enter a subject ID."); break
            p = cached_process(data, **o)
            if p.n < 12:
                weak.append(fname)
            pid = db.enroll(p, sid, name, finger, fname, o["ppi"], notes, case["examiner"])
            added += pid is not None
            skipped += pid is None
            bar.progress((k + 1) / len(items), f"{fname}")
        st.success(f"Enrolled {added} print(s). {skipped} duplicate(s) skipped.")
        if weak:
            st.warning(f"{len(weak)} print(s) have fewer than 12 minutiae and may not be searchable: "
                       + ", ".join(weak[:10]))

# ---------------------------------------------------------------------------- database
elif page == "Database":
    st.header("Reference database")
    rows = db.list()
    if not rows:
        st.info("No prints enrolled yet.")
    else:
        df = pd.DataFrame(rows)
        st.dataframe(df.drop(columns=["sha256"]), hide_index=True)
        st.download_button("Export list (CSV)", df.to_csv(index=False), "fpmatch_database.csv", "text/csv")
        c1, c2 = st.columns([1, 2])
        pid = c1.number_input("Print ID", min_value=int(df.id.min()), max_value=int(df.id.max()), value=int(df.id.min()))
        rec = db.get(int(pid))
        if rec:
            g = db.thumbnail(int(pid))
            c2.image(draw_template(g, rec["template"]), channels="BGR",
                     caption=f"{rec['subject_id']} · {rec['finger']} · {rec['n_minutiae']} minutiae", width=420)
            c1.code(f"SHA-256\n{rec['sha256']}", language=None)
            if c1.button("🗑 Delete this print"):
                db.delete(int(pid), case["examiner"]); st.rerun()

# ---------------------------------------------------------------------------- audit
elif page == "Audit log":
    st.header("Audit log")
    st.write("Every enrolment, deletion, comparison and search is recorded with UTC time and file hashes.")
    rows = db.audit_rows(2000)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True)
        st.download_button("Export audit log (CSV)", df.to_csv(index=False), "fpmatch_audit.csv", "text/csv")
    else:
        st.info("Nothing logged yet.")

# ---------------------------------------------------------------------------- help
else:
    st.header("How to read the results")
    st.markdown("""
**Score (0-100)**: how much of the overlapping area agrees, weighting each paired minutia by how well its
neighbouring minutiae also agree. **Supported pairs**: paired minutiae whose neighbourhoods agree. Count these
as corresponding points, not the raw *paired* count.

| Indication | Rule | Meaning |
|---|---|---|
| 🟢 Strong candidate | score ≥ 25 **and** ≥ 7 supported | Prioritise for examiner comparison |
| 🟠 Possible candidate | score ≥ 15 **or** ≥ 7 supported | Worth a look, weak evidence |
| ⚪ Not supported | below both | Minutiae do not support a common source |

These bands were set on a synthetic test set (no impostor exceeded score 17 or 6 supported pairs).
**Re-check them on your own data** (FVC / SOCOFing / lab prints) using `evaluate.py`.

**Tips for latent prints**
* Switch on *Latent / partial print*. Crop to the ridge area so background texture is ignored.
* Set the true resolution: a photo with a scale at 1000 ppi must be entered as 1000.
* Powder on dark surfaces: use *Invert*.
* Aim for at least 12 minutiae. Below that, any result is weak.

**This is a screening tool.** It ranks candidates for a human examiner (ACE-V). It does not identify anyone.
""")
