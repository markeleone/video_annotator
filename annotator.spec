from PyInstaller.utils.hooks import collect_all

block_cipher = None

a = Analysis(
    ['run_annotator.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
        'cv2', 'matplotlib', 'matplotlib.backends.backend_agg',
        'openpyxl', 'openpyxl.chart', 'openpyxl.styles', 'openpyxl.utils',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['torch', 'sam2', 'hydra', 'timm', 'decord'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CTAG Annotator',
    debug=False,
    strip=False,
    upx=False,
    console=False,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False,
    name='CTAG Annotator',
)

import sys as _sys
if _sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='CTAG Annotator.app',
        icon=None,
        bundle_identifier='com.ctag.annotator',
        info_plist={
            'CFBundleDisplayName': 'CTAG Annotator',
            'CFBundleShortVersionString': '1.0.0',
            'NSHighResolutionCapable': True,
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
        },
    )
