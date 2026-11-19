# PyInstaller spec for FP-Match.
# Build with packaging/build_mac.sh (macOS) or packaging\build_windows.bat (Windows).
# PyInstaller cannot cross-compile: build the Mac app on a Mac and the Windows app on Windows
# (or let .github/workflows/build.yml build both).
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
VERSION = re.search(r'__version__ = "(.+?)"', open(os.path.join(ROOT, "fpmatch", "__init__.py")).read()).group(1)

# Streamlit runs app.py as a script, so it ships as a data file. Its imports are not seen by
# PyInstaller's analysis and are listed as hidden imports instead.
datas = [(os.path.join(ROOT, "app.py"), "."),
         (os.path.join(ROOT, "samples"), "samples")]
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")
FPMATCH_MODULES = ["fpmatch." + f[:-3] for f in sorted(os.listdir(os.path.join(ROOT, "fpmatch")))
                   if f.endswith(".py") and f != "__init__.py"]
hiddenimports = ["fpmatch"] + FPMATCH_MODULES + ["pandas", "cv2", "scipy.ndimage", "skimage.morphology"]
hiddenimports += collect_submodules("streamlit", filter=lambda name: not name.startswith("streamlit.external"))

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "IPython", "pytest", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="FP-Match",
    console=sys.platform == "win32",  # Windows: a console window shows status; closing it stops the app
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="FP-Match", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="FP-Match.app",
        bundle_identifier="local.fpmatch.app",
        version=VERSION,
        info_plist={
            "CFBundleShortVersionString": VERSION,
            "CFBundleDisplayName": "FP-Match",
            "LSUIElement": True,  # no Dock icon: the UI is the browser tab; stop it with "Quit FP-Match"
            "NSHighResolutionCapable": True,
        },
    )
