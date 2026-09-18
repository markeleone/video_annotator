#!/usr/bin/env python3
"""
CTAG Annotator — Scene label annotation for underwater video.

Stripped-down version of EdgeTAM Feature Tracker.  All torch / SAM2 / hydra /
timm dependencies have been removed.  This file contains only:
  - Scene label (behavior + habitat) annotation bars
  - Read-only feature list (features loaded from a saved JSON, if any)
  - Video playback and frame seeking
  - Bottom timeline scrubber
  - JSON export and matplotlib analysis plots

Usage:
    python run_annotator.py video.mp4
    python run_annotator.py                      # opens file-picker dialog
    python run_annotator.py /path/to/frames/     # --frames-dir mode
    python run_annotator.py /path/ --frames-dir
"""

import sys
import os
import re
import csv
import copy
import json
import hashlib
import argparse
import shutil
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Qt compatibility: prefer PyQt6, fall back to PyQt5
# --------------------------------------------------------------------------
try:
    from PyQt6.QtCore import Qt, QTimer, QSize, QPoint, QRect, pyqtSignal
    from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont,
                             QShortcut, QKeySequence, QPolygon)
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QMessageBox, QListWidget,
        QListWidgetItem, QGroupBox, QLineEdit, QComboBox, QToolButton,
        QFrame, QSizePolicy, QProgressDialog, QSpinBox, QCompleter,
        QScrollArea, QCheckBox, QDialog, QTabWidget, QInputDialog, QLayout,
    )
    _QT6 = True
except ImportError:
    from PyQt5.QtCore import Qt, QTimer, QSize, QPoint, QRect, pyqtSignal
    from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont,
                             QShortcut, QKeySequence, QPolygon)
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
        QVBoxLayout, QFileDialog, QMessageBox, QListWidget,
        QListWidgetItem, QGroupBox, QLineEdit, QComboBox, QToolButton,
        QFrame, QSizePolicy, QProgressDialog, QSpinBox, QCompleter,
        QScrollArea, QCheckBox, QDialog, QTabWidget, QInputDialog, QLayout,
    )
    _QT6 = False

# --------------------------------------------------------------------------
# Modern UI constants
# --------------------------------------------------------------------------

# Behavior is split into two INDEPENDENT axes so a shark can be doing a
# movement (e.g. Foraging) *and* a social interaction (e.g. Parallel swimming)
# at the same time.  Both axes persist-until-changed, exactly like habitat.
# These lists are only the *defaults* used to seed behaviors.csv on first run;
# at runtime the app uses whatever is in the CSV (see load_behaviors()).
MOVEMENT_OPTIONS_DEFAULT = [
    "Burst swimming",
    "Low-speed swimming",
    "Stationary",
    "Foraging",
    "Turn",
]

SOCIAL_OPTIONS_DEFAULT = [
    "Solo",
    "Parallel swimming",
    "Following a shark",
    "Shark following tagged",
    "Brief interaction",
]

# Labels retired from the pickers. They are removed from the button bars (and
# from an existing behaviors.csv) but NEVER stripped from saved annotations —
# older deployments legitimately used them, so the data stays intact and the
# timeline/plots still render those segments.
RETIRED_SOCIAL = {"With 1+ sharks"}   # superseded by the Social-bar shark log

# Runtime lists (populated from CSV in main(); fall back to the defaults).
MOVEMENT_OPTIONS = list(MOVEMENT_OPTIONS_DEFAULT)
SOCIAL_OPTIONS = list(SOCIAL_OPTIONS_DEFAULT)

# Water visibility — its own persist-until-changed axis (5-point scale).
VISIBILITY_OPTIONS = ["Very bad", "Bad", "Fine", "Good", "Very good"]
# Values from the original 3-point scale still appear in older student files.
VISIBILITY_LEGACY_MAP = {"Poor": "Bad", "Moderate": "Fine"}

# Per-feature metadata option lists
CONFIDENCE_OPTIONS = ["", "High", "Medium", "Low"]
SEX_OPTIONS = ["", "Male", "Female", "Unknown"]

# Habitats are user-editable (like behaviors/species). These are the defaults
# used to seed habitats.csv on first run; runtime uses whatever's in the CSV.
HABITAT_OPTIONS_DEFAULT = ["Mangrove", "Rocky reef", "Sandy bottom", "Gravel", "Mud"]
HABITAT_OPTIONS = list(HABITAT_OPTIONS_DEFAULT)

SPECIES_CATEGORIES = {
    "Shark": [
        # Species
        "Lemon shark (Negaprion brevirostris)",
        "Pacific nurse shark (Ginglymostoma unami)",
        # Genus
        "Ginglymostoma sp.",
        "Negaprion sp.",
        # Family
        "Carcharhinidae",
        "Ginglymostomatidae",
    ],
    "Ray": [
        # Species
        "Longtail stingray (Hypanus longus)",
        "Pacific chupare stingray (Styracura pacifica)",
        "Longtail butterfly ray (Gymnura crebipunctata)",
        "Munk's devil ray (Mobula munkiana)",
        "Pacific eagle ray (Aetobatus laticeps)",
        "Speckled guitarfish (Pseudobatos glaucostygma)",
        "Prahl's guitarfish (Pseudobatos prahli)",
        "Pacific cownose ray (Rhinoptera steindachneri)",
        "Leopard round ray (Urobatis pardalis)",
        "Chilean round ray (Urotrygon chilensis)",
        # Genus
        "Aetobatus sp.",
        "Gymnura sp.",
        "Hypanus sp.",
        "Mobula sp.",
        "Pseudobatos sp.",
        "Rhinoptera sp.",
        "Styracura sp.",
        "Urobatis sp.",
        "Urotrygon sp.",
        # Family
        "Dasyatidae",
        "Gymnuridae",
        "Mobulidae",
        "Myliobatidae",
        "Rhinobatidae",
        "Rhinopteridae",
        "Urotrygonidae",
    ],
    "Teleost fish": [
        # Species
        "Convict surgeonfish (Acanthurus triostegus)",
        "Yellowfin surgeonfish (Acanthurus xanthopterus)",
        "Razor surgeonfish (Prionurus laticlavius)",
        "Dovi's cardinalfish (Apogon dovii)",
        "Chinese trumpetfish (Aulostomus chinensis)",
        "Finescale triggerfish (Balistes polylepis)",
        "Stone triggerfish (Pseudobalistes naufragium)",
        "Orangeside triggerfish (Sufflamen verres)",
        "Pacific agujon needlefish (Tylosurus fodiator)",
        "Large-banded blenny (Ophioblennius steindachneri)",
        "Sabertooth blenny (Plagiotremus azaleus)",
        "Leopard flounder (Bothus leopardinus)",
        "African pompano (Alectis ciliaris)",
        "Green jack (Caranx caballus)",
        "Pacific crevalle jack (Caranx caninus)",
        "Bigeye trevally (Caranx sexfasciatus)",
        "Rainbow runner (Elagatis bipinnulata)",
        "Golden trevally (Gnathanodon speciosus)",
        "Almaco jack (Seriola rivoliana)",
        "Paita pompano (Trachinotus paitensis)",
        "Blackblotch pompano (Trachinotus kennedyi)",
        "Gafftopsail pompano (Trachinotus rhodopus)",
        "Blackfin snook (Centropomus medius)",
        "Roughcheek barnacle blenny (Acanthemblemaria exilispinus)",
        "Hancock's blenny (Acanthemblemaria hancocki)",
        "Delta pikeblenny (Chaenopsis deltarrhis)",
        "Threebanded butterflyfish (Chaetodon humeralis)",
        "Barberfish (Johnrandallia nigrirostris)",
        "Giant hawkfish (Cirrhitus rivulatus)",
        "Coral hawkfish (Cirrhitichthys oxycephalus)",
        "Dolphinfish / mahi-mahi (Coryphaena hippurus)",
        "Spotfin burrfish (Chilomycterus reticulatus)",
        "Balloonfish (Diodon holocanthus)",
        "Spotted porcupinefish (Diodon hystrix)",
        "Pacific ladyfish (Elops affinis)",
        "Pacific spadefish (Chaetodipterus zonatus)",
        "Bluespotted cornetfish (Fistularia commersonii)",
        "Pacific flagfin mojarra (Eucinostomus currani)",
        "Slender mojarra (Gerres simillimus)",
        "Redlight goby (Coryphopterus urospilus)",
        "Spotted cleaner goby (Elacatinus puncticulatus)",
        "Burrito grunt (Anisotremus interruptus)",
        "Panamic porkfish (Anisotremus taeniatus)",
        "Yellowspotted grunt (Haemulon flaviguttatum)",
        "Spottail grunt (Haemulon maculicauda)",
        "Scudder's grunt (Haemulon scudderii)",
        "Greybar grunt (Haemulon sexfasciatum)",
        "Chere-chere grunt (Haemulon steindachneri)",
        "Goldeneye grunt (Microlepidotus brevipinnis)",
        "Panamic soldierfish (Myripristis leiognathus)",
        "Tinsel squirrelfish (Sargocentron suborbitale)",
        "Cortez sea chub (Kyphosus elegans)",
        "Blue-bronze sea chub (Kyphosus ocyurus)",
        "Mexican hogfish (Bodianus diplotaenia)",
        "Wounded wrasse (Halichoeres chierchiae)",
        "Chameleon wrasse (Halichoeres dispilus)",
        "Spinster wrasse (Halichoeres nicholsi)",
        "Bicolor wrasse (Halichoeres notospilus)",
        "Peacock razorfish (Iniistius pavo)",
        "Rockmover wrasse (Novaculichthys taeniourus)",
        "Sunset wrasse (Thalassoma grammaticum)",
        "Cortez rainbow wrasse (Thalassoma lucasanum)",
        "Panamic fanged blenny (Malacoctenus sudensis)",
        "Barred pargo (Hoplopagrus guentherii)",
        "Mullet snapper (Lutjanus aratus)",
        "Yellow snapper (Lutjanus argentiventris)",
        "Colorado snapper (Lutjanus colorado)",
        "Spotted rose snapper (Lutjanus guttatus)",
        "Golden snapper (Lutjanus inermis)",
        "Pacific dog snapper (Lutjanus novemfasciatus)",
        "Shortjaw tilefish (Malacanthus brevirostris)",
        "Unicorn filefish (Aluterus monoceros)",
        "Scrawled filefish (Aluterus scriptus)",
        "White mullet (Mugil curema)",
        "Mexican goatfish (Mulloidichthys dentatus)",
        "Bigscale goatfish (Pseudupeneus grandisquamis)",
        "Snowflake moray (Echidna nebulosa)",
        "Freckled moray (Echidna nocturna)",
        "Zebra moray (Gymnomuraena zebra)",
        "Chestnut moray (Gymnothorax castaneus)",
        "Dovi's moray (Gymnothorax dovii)",
        "Yellowmargin moray (Gymnothorax flavimarginatus)",
        "Panamic green moray (Gymnothorax panamensis)",
        "Undulated moray (Gymnothorax undulatus)",
        "Argus moray (Muraena argus)",
        "Hourglass moray (Muraena clepsydra)",
        "Jeweled moray (Muraena lentiginosa)",
        "Tiger reef-eel (Scuticaria tigrina)",
        "Roosterfish (Nematistius pectoralis)",
        "Tiger snake eel (Myrichthys tigrinus)",
        "Pacific snake eel (Ophichthus triseralis)",
        "Notched-fin snake eel (Quassiremus nothochir)",
        "Whitespotted boxfish (Ostracion meleagris)",
        "King angelfish (Holacanthus passer)",
        "Cortez angelfish (Pomacanthus zonipectus)",
        "Night sergeant (Abudefduf concolor)",
        "Panamic sergeant major (Abudefduf troschelii)",
        "Scissortail chromis (Azurina atrilobata)",
        "Bumphead damselfish (Microspathodon bairdii)",
        "Giant damselfish (Microspathodon dorsalis)",
        "Acapulco damselfish (Stegastes acapulcoensis)",
        "Beaubrummel (Stegastes flavilatus)",
        "Blue-barred parrotfish (Scarus ghobban)",
        "Greenblotch parrotfish (Scarus perrico)",
        "Ember parrotfish (Scarus rubroviolaceus)",
        "Black skipjack (Euthynnus lineatus)",
        "Pacific sierra (Scomberomorus sierra)",
        "Stone scorpionfish (Scorpaena mystes)",
        "Pacific mutton hamlet (Alphestes immaculatus)",
        "Pacific graysby (Cephalopholis colonus)",
        "Creolefish (Paranthias panamensis)",
        "Leather bass (Dermatolepis dermatolepis)",
        "Spotted grouper (Epinephelus analogus)",
        "Starry grouper (Epinephelus labriformis)",
        "Pacific goliath grouper (Epinephelus quinquefasciatus)",
        "Broomtail grouper (Mycteroperca xenarcha)",
        "Mottled soapfish (Rypticus bicolor)",
        "Flag serrano (Serranus psittacinus)",
        "Pacific porgy (Calamus brachysomus)",
        "Mexican barracuda (Sphyraena ensis)",
        "Bluestripe pipefish (Doryrhamphus excisus)",
        "Calico lizardfish (Synodus lacertinus)",
        "Whitespotted puffer (Arothron hispidus)",
        "Guineafowl puffer (Arothron meleagris)",
        "Spotted sharpnose puffer (Canthigaster punctatissima)",
        "Bullseye puffer (Sphoeroides annulatus)",
        "Lobed puffer (Sphoeroides lobatus)",
        "Lucilla's triplefin (Axoclinus lucillae)",
        # Genus
        "Abudefduf sp.",
        "Acanthemblemaria sp.",
        "Acanthurus sp.",
        "Alectis sp.",
        "Alphestes sp.",
        "Aluterus sp.",
        "Anisotremus sp.",
        "Apogon sp.",
        "Arothron sp.",
        "Aulostomus sp.",
        "Axoclinus sp.",
        "Azurina sp.",
        "Balistes sp.",
        "Bodianus sp.",
        "Bothus sp.",
        "Calamus sp.",
        "Canthigaster sp.",
        "Caranx sp.",
        "Centropomus sp.",
        "Cephalopholis sp.",
        "Chaenopsis sp.",
        "Chaetodipterus sp.",
        "Chaetodon sp.",
        "Chilomycterus sp.",
        "Cirrhitichthys sp.",
        "Cirrhitus sp.",
        "Coryphaena sp.",
        "Coryphopterus sp.",
        "Dermatolepis sp.",
        "Diodon sp.",
        "Doryrhamphus sp.",
        "Echidna sp.",
        "Elacatinus sp.",
        "Elagatis sp.",
        "Elops sp.",
        "Epinephelus sp.",
        "Eucinostomus sp.",
        "Euthynnus sp.",
        "Fistularia sp.",
        "Gerres sp.",
        "Gnathanodon sp.",
        "Gymnomuraena sp.",
        "Gymnothorax sp.",
        "Haemulon sp.",
        "Halichoeres sp.",
        "Holacanthus sp.",
        "Hoplopagrus sp.",
        "Iniistius sp.",
        "Johnrandallia sp.",
        "Kyphosus sp.",
        "Lutjanus sp.",
        "Malacanthus sp.",
        "Malacoctenus sp.",
        "Microlepidotus sp.",
        "Microspathodon sp.",
        "Mugil sp.",
        "Mulloidichthys sp.",
        "Muraena sp.",
        "Myrichthys sp.",
        "Mycteroperca sp.",
        "Myripristis sp.",
        "Nematistius sp.",
        "Novaculichthys sp.",
        "Ophioblennius sp.",
        "Ophichthus sp.",
        "Ostracion sp.",
        "Paranthias sp.",
        "Plagiotremus sp.",
        "Pomacanthus sp.",
        "Prionurus sp.",
        "Pseudobalistes sp.",
        "Pseudupeneus sp.",
        "Quassiremus sp.",
        "Rypticus sp.",
        "Sargocentron sp.",
        "Scarus sp.",
        "Scomberomorus sp.",
        "Scorpaena sp.",
        "Scuticaria sp.",
        "Seriola sp.",
        "Serranus sp.",
        "Sphoeroides sp.",
        "Sphyraena sp.",
        "Stegastes sp.",
        "Sufflamen sp.",
        "Synodus sp.",
        "Thalassoma sp.",
        "Trachinotus sp.",
        "Tylosurus sp.",
        # Family
        "Acanthuridae",
        "Apogonidae",
        "Aulostomidae",
        "Balistidae",
        "Belonidae",
        "Blenniidae",
        "Bothidae",
        "Carangidae",
        "Centropomidae",
        "Chaenopsidae",
        "Chaetodontidae",
        "Cirrhitidae",
        "Coryphaenidae",
        "Diodontidae",
        "Elopidae",
        "Ephippidae",
        "Fistulariidae",
        "Gerreidae",
        "Gobiidae",
        "Haemulidae",
        "Holocentridae",
        "Kyphosidae",
        "Labridae",
        "Labrisomidae",
        "Lutjanidae",
        "Malacanthidae",
        "Monacanthidae",
        "Mugilidae",
        "Mullidae",
        "Muraenidae",
        "Nematistiidae",
        "Ophichthidae",
        "Ostraciidae",
        "Pomacanthidae",
        "Pomacentridae",
        "Scaridae",
        "Scombridae",
        "Scorpaenidae",
        "Serranidae",
        "Sparidae",
        "Sphyraenidae",
        "Syngnathidae",
        "Synodontidae",
        "Tetraodontidae",
        "Tripterygiidae",
    ],
    "Sea turtle": [
        # Species
        "Hawksbill sea turtle (Eretmochelys imbricata)",
        "Olive ridley sea turtle (Lepidochelys olivacea)",
        "Green sea turtle (Chelonia mydas)",
        # Genus
        "Chelonia sp.",
        "Eretmochelys sp.",
        "Lepidochelys sp.",
        # Family
        "Cheloniidae",
    ],
    "Other": [],
}
# Snapshot of the built-in taxonomy, used to seed species.csv on first run.
SPECIES_CATEGORIES_DEFAULT = {cat: list(sps) for cat, sps in SPECIES_CATEGORIES.items()}
# Category display order (also the Category dropdown order).
SPECIES_CATEGORY_ORDER = list(SPECIES_CATEGORIES.keys())
ALL_SPECIES = [s for species in SPECIES_CATEGORIES.values() for s in species]


# --------------------------------------------------------------------------
# User-editable label config (species + behaviors) — no code edits needed
# --------------------------------------------------------------------------
# On first launch we seed CSV files under ~/CTAG_Annotator from the built-in
# defaults.  Researchers can edit those CSVs directly, or use the in-app
# "＋ Add" buttons which append to them live.

CONFIG_DIR = os.path.join(os.path.expanduser("~"), "CTAG_Annotator")
# Shown in Help and written into every error log line, so a bug report says
# which build the annotator was running (students keep old zips around).
APP_VERSION = "2026.09.18"
LOG_DIR = os.path.join(CONFIG_DIR, "logs")
SPECIES_CSV = os.path.join(CONFIG_DIR, "species.csv")
TAXONOMY_CSV = os.path.join(CONFIG_DIR, "taxonomy.csv")
BEHAVIORS_CSV = os.path.join(CONFIG_DIR, "behaviors.csv")
HABITATS_CSV = os.path.join(CONFIG_DIR, "habitats.csv")
# Guaranteed-local place for annotations/autosaves — lives in the user's home
# (NOT inside any cloud-synced mount), so saves work even when the *video* is
# streamed from Google Drive / iCloud / Dropbox / OneDrive.
LOCAL_ANNOTATIONS_DIR = os.path.join(CONFIG_DIR, "annotations")

# Substrings that indicate a path lives inside a cloud "file provider" mount,
# where app writes can silently fail or not sync (the common cause of
# "my JSON didn't save to the folder").
_CLOUD_PATH_MARKERS = (
    "/Library/CloudStorage/",   # Google Drive, OneDrive, Dropbox, Box (modern macOS)
    "com.apple.CloudDocs",      # iCloud Drive
    "/Google Drive",            # legacy Google Drive for Desktop mount
    "GoogleDrive-",
    "/Dropbox",
    "/OneDrive",
    "/My Drive",
    "/Shared drives",
)


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def source_key_for(path: str) -> str:
    """Stable, filesystem-safe id for a source path (shared by autosave and the
    startup folder scan, so both agree on which video an autosave belongs to)."""
    base = path or "session"
    stem = os.path.splitext(os.path.basename(base.rstrip("/")))[0] or "session"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:60]
    h = hashlib.sha1(base.encode("utf-8", "ignore")).hexdigest()[:8]
    return f"{stem}_{h}"


def autosave_path_for(path: str) -> str:
    return os.path.join(LOCAL_ANNOTATIONS_DIR, source_key_for(path) + ".autosave.json")


def is_annotated(path: str) -> bool:
    """True if this video already has saved annotation work."""
    p = autosave_path_for(path)
    try:
        return os.path.exists(p) and os.path.getsize(p) > 0
    except OSError:
        return False


def list_videos_in_folder(folder: str):
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names
            if n.lower().endswith(VIDEO_EXTS) and not n.startswith("._")]


def is_cloud_path(path: str) -> bool:
    """True if `path` appears to live inside a cloud-synced virtual filesystem."""
    return any(m in (path or "") for m in _CLOUD_PATH_MARKERS)


# --------------------------------------------------------------------------
# Taxonomy — genus → family, so IDs can be recorded at family, genus or
# species level and the three pickers can fill each other in.
# --------------------------------------------------------------------------
GENUS_FAMILY = {
    # Sharks
    "Ginglymostoma": "Ginglymostomatidae", "Negaprion": "Carcharhinidae",
    # Rays
    "Aetobatus": "Myliobatidae", "Gymnura": "Gymnuridae", "Hypanus": "Dasyatidae",
    "Mobula": "Mobulidae", "Pseudobatos": "Rhinobatidae",
    "Rhinoptera": "Rhinopteridae", "Styracura": "Dasyatidae",
    "Urobatis": "Urotrygonidae", "Urotrygon": "Urotrygonidae",
    # Sea turtles
    "Chelonia": "Cheloniidae", "Eretmochelys": "Cheloniidae",
    "Lepidochelys": "Cheloniidae",
    # Teleosts
    "Abudefduf": "Pomacentridae", "Acanthemblemaria": "Chaenopsidae",
    "Acanthurus": "Acanthuridae", "Alectis": "Carangidae",
    "Alphestes": "Serranidae", "Aluterus": "Monacanthidae",
    "Anisotremus": "Haemulidae", "Apogon": "Apogonidae",
    "Arothron": "Tetraodontidae", "Aulostomus": "Aulostomidae",
    "Axoclinus": "Tripterygiidae", "Azurina": "Pomacentridae",
    "Balistes": "Balistidae", "Bodianus": "Labridae", "Bothus": "Bothidae",
    "Calamus": "Sparidae", "Canthigaster": "Tetraodontidae",
    "Caranx": "Carangidae", "Centropomus": "Centropomidae",
    "Cephalopholis": "Serranidae", "Chaenopsis": "Chaenopsidae",
    "Chaetodipterus": "Ephippidae", "Chaetodon": "Chaetodontidae",
    "Chilomycterus": "Diodontidae", "Cirrhitichthys": "Cirrhitidae",
    "Cirrhitus": "Cirrhitidae", "Coryphaena": "Coryphaenidae",
    "Coryphopterus": "Gobiidae", "Dermatolepis": "Serranidae",
    "Diodon": "Diodontidae", "Doryrhamphus": "Syngnathidae",
    "Echidna": "Muraenidae", "Elacatinus": "Gobiidae",
    "Elagatis": "Carangidae", "Elops": "Elopidae",
    "Epinephelus": "Serranidae", "Eucinostomus": "Gerreidae",
    "Euthynnus": "Scombridae", "Fistularia": "Fistulariidae",
    "Gerres": "Gerreidae", "Gnathanodon": "Carangidae",
    "Gymnomuraena": "Muraenidae", "Gymnothorax": "Muraenidae",
    "Haemulon": "Haemulidae", "Halichoeres": "Labridae",
    "Holacanthus": "Pomacanthidae", "Hoplopagrus": "Lutjanidae",
    "Iniistius": "Labridae", "Johnrandallia": "Chaetodontidae",
    "Kyphosus": "Kyphosidae", "Lutjanus": "Lutjanidae",
    "Malacanthus": "Malacanthidae", "Malacoctenus": "Labrisomidae",
    "Microlepidotus": "Haemulidae", "Microspathodon": "Pomacentridae",
    "Mugil": "Mugilidae", "Mulloidichthys": "Mullidae",
    "Muraena": "Muraenidae", "Mycteroperca": "Serranidae",
    "Myrichthys": "Ophichthidae", "Myripristis": "Holocentridae",
    "Nematistius": "Nematistiidae", "Novaculichthys": "Labridae",
    "Ophichthus": "Ophichthidae", "Ophioblennius": "Blenniidae",
    "Ostracion": "Ostraciidae", "Paranthias": "Serranidae",
    "Plagiotremus": "Blenniidae", "Pomacanthus": "Pomacanthidae",
    "Prionurus": "Acanthuridae", "Pseudobalistes": "Balistidae",
    "Pseudupeneus": "Mullidae", "Quassiremus": "Ophichthidae",
    "Rypticus": "Serranidae", "Sargocentron": "Holocentridae",
    "Scarus": "Scaridae", "Scomberomorus": "Scombridae",
    "Scorpaena": "Scorpaenidae", "Scuticaria": "Muraenidae",
    "Seriola": "Carangidae", "Serranus": "Serranidae",
    "Sphoeroides": "Tetraodontidae", "Sphyraena": "Sphyraenidae",
    "Stegastes": "Pomacentridae", "Sufflamen": "Balistidae",
    "Synodus": "Synodontidae", "Thalassoma": "Labridae",
    "Trachinotus": "Carangidae", "Tylosurus": "Belonidae",
}

# Plain-English names for the families, so an annotator who knows "groupers"
# but not "Serranidae" can still find and recognise the right family.
FAMILY_COMMON = {
    # Sharks / rays / turtles
    "Carcharhinidae": "requiem sharks", "Ginglymostomatidae": "nurse sharks",
    "Dasyatidae": "whiptail stingrays", "Gymnuridae": "butterfly rays",
    "Mobulidae": "devil rays & mantas", "Myliobatidae": "eagle rays",
    "Rhinobatidae": "guitarfishes", "Rhinopteridae": "cownose rays",
    "Urotrygonidae": "round stingrays", "Cheloniidae": "sea turtles",
    # Teleosts
    "Acanthuridae": "surgeonfishes", "Apogonidae": "cardinalfishes",
    "Aulostomidae": "trumpetfishes", "Balistidae": "triggerfishes",
    "Belonidae": "needlefishes", "Blenniidae": "combtooth blennies",
    "Bothidae": "lefteye flounders", "Carangidae": "jacks & pompanos",
    "Centropomidae": "snooks", "Chaenopsidae": "tube blennies",
    "Chaetodontidae": "butterflyfishes", "Cirrhitidae": "hawkfishes",
    "Coryphaenidae": "dolphinfishes", "Diodontidae": "porcupinefishes",
    "Elopidae": "ladyfishes", "Ephippidae": "spadefishes",
    "Fistulariidae": "cornetfishes", "Gerreidae": "mojarras",
    "Gobiidae": "gobies", "Haemulidae": "grunts",
    "Holocentridae": "squirrelfishes", "Kyphosidae": "sea chubs",
    "Labridae": "wrasses", "Labrisomidae": "labrisomid blennies",
    "Lutjanidae": "snappers", "Malacanthidae": "tilefishes",
    "Monacanthidae": "filefishes", "Mugilidae": "mullets",
    "Mullidae": "goatfishes", "Muraenidae": "moray eels",
    "Nematistiidae": "roosterfish", "Ophichthidae": "snake eels",
    "Ostraciidae": "boxfishes", "Pomacanthidae": "angelfishes",
    "Pomacentridae": "damselfishes", "Scaridae": "parrotfishes",
    "Scombridae": "mackerels & tunas", "Scorpaenidae": "scorpionfishes",
    "Serranidae": "groupers & sea basses", "Sparidae": "porgies",
    "Sphyraenidae": "barracudas", "Syngnathidae": "pipefishes & seahorses",
    "Synodontidae": "lizardfishes", "Tetraodontidae": "puffers",
    "Tripterygiidae": "triplefins",
}


def family_display(fam):
    """'Serranidae' -> 'Serranidae — groupers & sea basses'."""
    if not fam:
        return ""
    common = FAMILY_COMMON.get(fam)
    return f"{fam} — {common}" if common else fam


def family_from_display(text):
    """Inverse of family_display; tolerant of a bare family name."""
    return (text or "").split(" — ")[0].strip()


_SPECIES_RE = re.compile(r"^(.*?)\s*\(([A-Z][a-z]+)\s+([a-z][a-z\-]*)\)\s*$")


def _build_default_taxonomy():
    """Expand the curated flat lists into structured taxonomy rows.

    Each row: category, family, genus, species (binomial), common.
    Blank genus/species means the entry only resolves to that higher rank.
    """
    rows = []
    for cat, items in SPECIES_CATEGORIES_DEFAULT.items():
        for it in items:
            m = _SPECIES_RE.match(it)
            if m:
                common, gen, epithet = m.group(1).strip(), m.group(2), m.group(3)
                rows.append({"category": cat, "family": GENUS_FAMILY.get(gen, ""),
                             "genus": gen, "species": f"{gen} {epithet}",
                             "common": common})
            elif it.endswith(" sp."):
                gen = it[:-4].strip()
                rows.append({"category": cat, "family": GENUS_FAMILY.get(gen, ""),
                             "genus": gen, "species": "", "common": ""})
            elif it:
                rows.append({"category": cat, "family": it,
                             "genus": "", "species": "", "common": ""})
    return rows


class Taxonomy:
    """Indexed taxonomy supporting family/genus/species lookups in any order."""

    FIELDS = ["category", "family", "genus", "species", "common"]

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.reindex()

    def reindex(self):
        self.categories = []
        self._fam_of_genus = {}
        self._cat_of_family = {}
        for r in self.rows:
            c = r.get("category") or "Other"
            if c not in self.categories:
                self.categories.append(c)
            if r.get("genus") and r.get("family"):
                self._fam_of_genus.setdefault(r["genus"], r["family"])
            if r.get("family"):
                self._cat_of_family.setdefault(r["family"], c)

    def _scope(self, category=None):
        if not category or category == "Other":
            return self.rows
        return [r for r in self.rows if r.get("category") == category]

    def families(self, category=None):
        seen = []
        for r in self._scope(category):
            f = r.get("family")
            if f and f not in seen:
                seen.append(f)
        return sorted(seen)

    def genera(self, category=None, family=None):
        seen = []
        for r in self._scope(category):
            if family and r.get("family") != family:
                continue
            g = r.get("genus")
            if g and g not in seen:
                seen.append(g)
        return sorted(seen)

    def species_rows(self, category=None, family=None, genus=None):
        out = []
        for r in self._scope(category):
            if not r.get("species"):
                continue
            if family and r.get("family") != family:
                continue
            if genus and r.get("genus") != genus:
                continue
            out.append(r)
        return sorted(out, key=lambda r: r["species"])

    def family_of_genus(self, genus):
        return self._fam_of_genus.get(genus, "")

    def category_of_family(self, family):
        return self._cat_of_family.get(family, "")

    def row_for_species(self, binomial):
        for r in self.rows:
            if r.get("species") == binomial:
                return r
        return None

    def search(self, text, category=None, limit=60):
        """Rank-tagged fuzzy search across family, genus, species and common
        name — this is the 'I already know what it is, let me type it' path."""
        q = (text or "").strip().lower()
        if not q:
            return []
        hits, seen = [], set()

        def add(kind, label, row, score):
            key = (kind, label)
            if key not in seen:
                seen.add(key)
                hits.append({"kind": kind, "label": label, "row": row, "score": score})

        for r in self._scope(category):
            sp, com, gen, fam = (r.get("species", ""), r.get("common", ""),
                                 r.get("genus", ""), r.get("family", ""))
            if sp:
                hay = f"{com} {sp}".lower()
                if q in hay:
                    add("species", f"{com} ({sp})" if com else sp, r,
                        0 if hay.startswith(q) else 1)
            if gen and q in gen.lower():
                add("genus", gen, r, 0 if gen.lower().startswith(q) else 1)
            if fam:
                fam_hay = f"{fam} {FAMILY_COMMON.get(fam, '')}".lower()
                if q in fam_hay:
                    add("family", family_display(fam), r,
                        0 if fam_hay.startswith(q) else 1)

        rank = {"species": 0, "genus": 1, "family": 2}
        hits.sort(key=lambda h: (h["score"], rank[h["kind"]], h["label"]))
        return hits[:limit]


TAXONOMY = Taxonomy()


def _ensure_config_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)


def _seed_species_csv():
    _ensure_config_dir()
    with open(SPECIES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "species"])
        for cat in SPECIES_CATEGORY_ORDER:
            sps = SPECIES_CATEGORIES_DEFAULT.get(cat, [])
            if sps:
                for s in sps:
                    w.writerow([cat, s])
            else:
                w.writerow([cat, ""])   # keep the category even with no species


def _seed_behaviors_csv():
    _ensure_config_dir()
    with open(BEHAVIORS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "label"])
        for lab in MOVEMENT_OPTIONS_DEFAULT:
            w.writerow(["movement", lab])
        for lab in SOCIAL_OPTIONS_DEFAULT:
            w.writerow(["social", lab])


def load_species_config():
    """Return an ordered {category: [species,...]} dict from species.csv.

    Seeds the CSV from the built-in taxonomy on first run.  Always keeps the
    standard categories present (in order) so the UI stays predictable.
    """
    if not os.path.exists(SPECIES_CSV):
        try:
            _seed_species_csv()
        except OSError:
            return {cat: list(sps) for cat, sps in SPECIES_CATEGORIES_DEFAULT.items()}

    cats: Dict[str, list] = {}
    try:
        with open(SPECIES_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cat = (row.get("category") or "").strip()
                sp = (row.get("species") or "").strip()
                if not cat:
                    continue
                cats.setdefault(cat, [])
                if sp and sp not in cats[cat]:
                    cats[cat].append(sp)
    except (OSError, csv.Error):
        return {cat: list(sps) for cat, sps in SPECIES_CATEGORIES_DEFAULT.items()}

    # Guarantee the standard categories exist and lead, preserving CSV order
    # for any extra categories the user added.
    ordered: Dict[str, list] = {}
    for cat in SPECIES_CATEGORY_ORDER:
        ordered[cat] = cats.pop(cat, [])
    ordered.update(cats)
    return ordered


def load_behaviors_config():
    """Return (movement_list, social_list) from behaviors.csv, seeding first run."""
    if not os.path.exists(BEHAVIORS_CSV):
        try:
            _seed_behaviors_csv()
        except OSError:
            return list(MOVEMENT_OPTIONS_DEFAULT), list(SOCIAL_OPTIONS_DEFAULT)

    movement, social = [], []
    try:
        with open(BEHAVIORS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                axis = (row.get("axis") or "").strip().lower()
                lab = (row.get("label") or "").strip()
                if not lab:
                    continue
                if axis == "movement" and lab not in movement:
                    movement.append(lab)
                elif axis == "social" and lab not in social:
                    social.append(lab)
    except (OSError, csv.Error):
        return list(MOVEMENT_OPTIONS_DEFAULT), list(SOCIAL_OPTIONS_DEFAULT)

    social = [s for s in social if s not in RETIRED_SOCIAL]
    return (movement or list(MOVEMENT_OPTIONS_DEFAULT),
            social or list(SOCIAL_OPTIONS_DEFAULT))


def write_taxonomy_csv(rows):
    _ensure_config_dir()
    with open(TAXONOMY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=Taxonomy.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in Taxonomy.FIELDS})


def load_taxonomy_config():
    """Load taxonomy.csv, seeding it from the curated defaults on first run."""
    if not os.path.exists(TAXONOMY_CSV):
        rows = _build_default_taxonomy()
        try:
            write_taxonomy_csv(rows)
        except OSError:
            pass
        return Taxonomy(rows)
    rows = []
    try:
        with open(TAXONOMY_CSV, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                r = {k: (row.get(k) or "").strip() for k in Taxonomy.FIELDS}
                if any(r[k] for k in ("family", "genus", "species")):
                    r["category"] = r["category"] or "Other"
                    if not r["family"] and r["genus"]:
                        r["family"] = GENUS_FAMILY.get(r["genus"], "")
                    rows.append(r)
    except (OSError, csv.Error):
        rows = _build_default_taxonomy()
    return Taxonomy(rows or _build_default_taxonomy())


def import_taxonomy_csv(path, replace=True):
    """Import a user taxonomy CSV.

    Accepts the full category/family/genus/species/common form, or simpler
    files: a single column of names, or family,genus,species. Anything the
    header doesn't name is inferred (genus from the binomial, family from
    the built-in genus->family table).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        raw = list(csv.reader(f))
    if not raw:
        raise ValueError("CSV is empty.")
    header = [h.strip().lower() for h in raw[0]]
    known = {"category", "family", "genus", "species", "common", "common_name"}
    has_header = any(h in known for h in header)
    rows = []
    if has_header:
        idx = {h: i for i, h in enumerate(header)}
        for line in raw[1:]:
            def get(name):
                i = idx.get(name)
                return (line[i].strip() if i is not None and i < len(line) else "")
            r = {"category": get("category") or "Other",
                 "family": get("family"), "genus": get("genus"),
                 "species": get("species"),
                 "common": get("common") or get("common_name")}
            rows.append(r)
    else:
        for line in raw:
            if not line:
                continue
            name = line[0].strip()
            if not name:
                continue
            rows.append({"category": "Other", "family": "", "genus": "",
                         "species": name, "common": ""})
    # normalise: infer genus from binomial, family from genus
    clean = []
    for r in rows:
        sp = r["species"]
        m = _SPECIES_RE.match(sp)
        if m:  # someone pasted "Common (Genus epithet)"
            r["common"] = r["common"] or m.group(1).strip()
            r["genus"] = r["genus"] or m.group(2)
            sp = f"{m.group(2)} {m.group(3)}"
        if sp and not r["genus"]:
            parts = sp.split()
            if len(parts) >= 2 and parts[0][:1].isupper():
                r["genus"] = parts[0]
        if not r["family"] and r["genus"]:
            r["family"] = GENUS_FAMILY.get(r["genus"], "")
        r["species"] = sp
        if any(r[k] for k in ("family", "genus", "species")):
            clean.append(r)
    if not clean:
        raise ValueError("No usable taxonomy rows found in the CSV.")
    if not replace:
        clean = load_taxonomy_config().rows + clean
    write_taxonomy_csv(clean)


def _seed_habitats_csv():
    _ensure_config_dir()
    with open(HABITATS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["habitat"])
        for h in HABITAT_OPTIONS_DEFAULT:
            w.writerow([h])


def load_habitats_config():
    """Return the habitat list from habitats.csv, seeding first run."""
    if not os.path.exists(HABITATS_CSV):
        try:
            _seed_habitats_csv()
        except OSError:
            return list(HABITAT_OPTIONS_DEFAULT)
    out = []
    try:
        with open(HABITATS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                h = (row.get("habitat") or "").strip()
                if h and h not in out:
                    out.append(h)
    except (OSError, csv.Error):
        return list(HABITAT_OPTIONS_DEFAULT)
    return out or list(HABITAT_OPTIONS_DEFAULT)


# ---- full-file writers (used by add / remove / import / restore) ----------

def write_species_csv(cats: Dict[str, list]):
    _ensure_config_dir()
    with open(SPECIES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "species"])
        for cat, sps in cats.items():
            if sps:
                for s in sps:
                    w.writerow([cat, s])
            else:
                w.writerow([cat, ""])


def write_behaviors_csv(movement: list, social: list):
    _ensure_config_dir()
    with open(BEHAVIORS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "label"])
        for lab in movement:
            w.writerow(["movement", lab])
        for lab in social:
            w.writerow(["social", lab])


def write_habitats_csv(habs: list):
    _ensure_config_dir()
    with open(HABITATS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["habitat"])
        for h in habs:
            w.writerow([h])


def append_species_to_csv(category: str, species: str):
    SPECIES_CATEGORIES.setdefault(category, [])
    if species not in SPECIES_CATEGORIES[category]:
        SPECIES_CATEGORIES[category].append(species)
    write_species_csv(SPECIES_CATEGORIES)


def append_behavior_to_csv(axis: str, label: str):
    lst = MOVEMENT_OPTIONS if axis == "movement" else SOCIAL_OPTIONS
    if label not in lst:
        lst.append(label)
    write_behaviors_csv(MOVEMENT_OPTIONS, SOCIAL_OPTIONS)


def import_species_csv(path: str, replace: bool = True):
    """Import a CSV of species. Accepts 'category,species' rows, or a single
    column of species names (imported under 'Other')."""
    imported: Dict[str, list] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("CSV is empty.")
    start = 0
    header = [c.strip().lower() for c in rows[0]]
    two_col = len(header) >= 2 and ("categor" in header[0])
    one_col_hdr = len(header) >= 1 and header[0] in ("species", "name")
    if two_col or one_col_hdr:
        start = 1
    for row in rows[start:]:
        if not row:
            continue
        if len(row) >= 2 and (two_col or not one_col_hdr):
            cat = (row[0] or "").strip() or "Other"
            sp = (row[1] or "").strip()
        else:
            cat, sp = "Other", (row[0] or "").strip()
        if not sp:
            continue
        imported.setdefault(cat, [])
        if sp not in imported[cat]:
            imported[cat].append(sp)
    if not imported:
        raise ValueError("No species rows found in the CSV.")
    global SPECIES_CATEGORIES
    if replace:
        merged = {c: [] for c in SPECIES_CATEGORY_ORDER}
        merged.update(imported)
    else:
        merged = {c: list(v) for c, v in SPECIES_CATEGORIES.items()}
        for c, sps in imported.items():
            merged.setdefault(c, [])
            for s in sps:
                if s not in merged[c]:
                    merged[c].append(s)
    write_species_csv(merged)


def import_behaviors_csv(path: str, replace: bool = True):
    """Import 'axis,label' rows (axis = movement|social)."""
    mov, soc = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for i, row in enumerate(rows):
        if not row:
            continue
        if i == 0 and len(row) >= 2 and row[0].strip().lower().startswith("axis"):
            continue
        axis = (row[0] or "").strip().lower() if len(row) >= 2 else ""
        lab = (row[1] if len(row) >= 2 else row[0]).strip()
        if not lab:
            continue
        if axis == "social":
            soc.append(lab)
        else:
            mov.append(lab)
    if not mov and not soc:
        raise ValueError("No behavior rows found in the CSV.")
    if replace:
        write_behaviors_csv(mov or list(MOVEMENT_OPTIONS),
                            soc or list(SOCIAL_OPTIONS))
    else:
        write_behaviors_csv(MOVEMENT_OPTIONS + [m for m in mov if m not in MOVEMENT_OPTIONS],
                            SOCIAL_OPTIONS + [s for s in soc if s not in SOCIAL_OPTIONS])


def import_habitats_csv(path: str, replace: bool = True):
    """Import a single column of habitat names (optional 'habitat' header)."""
    habs = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row:
                continue
            h = (row[0] or "").strip()
            if i == 0 and h.lower() == "habitat":
                continue
            if h and h not in habs:
                habs.append(h)
    if not habs:
        raise ValueError("No habitat rows found in the CSV.")
    if replace:
        write_habitats_csv(habs)
    else:
        write_habitats_csv(HABITAT_OPTIONS + [h for h in habs if h not in HABITAT_OPTIONS])


def restore_default(kind: str):
    """Restore one label category to the built-in defaults."""
    if kind == "species":
        write_species_csv({c: list(v) for c, v in SPECIES_CATEGORIES_DEFAULT.items()})
    elif kind == "behaviors":
        write_behaviors_csv(list(MOVEMENT_OPTIONS_DEFAULT), list(SOCIAL_OPTIONS_DEFAULT))
    elif kind == "habitats":
        write_habitats_csv(list(HABITAT_OPTIONS_DEFAULT))
    elif kind == "taxonomy":
        write_taxonomy_csv(_build_default_taxonomy())


def load_all_label_config():
    """Populate the runtime global label lists from the CSV config files."""
    global SPECIES_CATEGORIES, SPECIES_CATEGORY_ORDER, ALL_SPECIES
    global MOVEMENT_OPTIONS, SOCIAL_OPTIONS, HABITAT_OPTIONS, TAXONOMY
    SPECIES_CATEGORIES = load_species_config()
    SPECIES_CATEGORY_ORDER = list(SPECIES_CATEGORIES.keys())
    ALL_SPECIES = [s for sps in SPECIES_CATEGORIES.values() for s in sps]
    MOVEMENT_OPTIONS, SOCIAL_OPTIONS = load_behaviors_config()
    HABITAT_OPTIONS = load_habitats_config()
    TAXONOMY = load_taxonomy_config()
    for c in TAXONOMY.categories:
        if c not in SPECIES_CATEGORY_ORDER:
            SPECIES_CATEGORY_ORDER.append(c)


# Modern-ish palette (RGB) for tracked features
FEATURE_COLORS = [
    (16, 185, 129),   # emerald-500
    (245, 158, 11),   # amber-500
    (14, 165, 233),   # sky-500
    (217, 70, 239),   # fuchsia-500
    (132, 204, 22),   # lime-500
    (6, 182, 212),    # cyan-500
    (244, 63, 94),    # rose-500
    (34, 197, 94),    # green-500
]

APP_QSS = r"""
QMainWindow { background: #F6F7FB; }
* { font-family: "Inter","SF Pro Text","SF Pro Display","Segoe UI","Helvetica","Arial"; font-size: 12px; color: #0F172A; }

QFrame#Card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
}

QLabel#Title {
  font-size: 16px;
  font-weight: 600;
  color: #0F172A;
}

QLabel#Subtle {
  color: #64748B;
}

QLabel#BarTitle {
  color: #334155;
  font-weight: 600;
}

QLineEdit, QComboBox, QDoubleSpinBox {
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 8px;
  padding: 4px 8px;
}

/* NOTE: do NOT restyle QComboBox::drop-down / ::down-arrow here. Qt does not
   support the CSS border-triangle trick (it paints a solid block), and any
   drop-down rule without a valid arrow image makes Qt drop the native
   indicator — which left the boxes looking like plain text fields so nobody
   realised the lists were browsable. Fusion's own caret is drawn instead. */

/* Roomy, scrollable popup list */
QComboBox QAbstractItemView::item {
  min-height: 22px;
  padding: 3px 6px;
}

QCheckBox { spacing: 10px; }
QRadioButton { spacing: 8px; }

QPushButton {
  background: #111827;
  color: #FFFFFF;
  border: none;
  border-radius: 9px;
  padding: 5px 10px;
  font-weight: 600;
}
QPushButton:hover { background: #0B1220; }
QPushButton:pressed { background: #030712; }
QPushButton:disabled { background: #9CA3AF; color: #F9FAFB; }

QPushButton#Secondary {
  background: #EEF2FF;
  color: #1E293B;
  border: 1px solid #C7D2FE;
}
QPushButton#Secondary:hover { background: #E0E7FF; }

QToolButton[segmented="true"] {
  background: #F1F5F9;
  border: 1px solid #E2E8F0;
  border-radius: 999px;
  padding: 3px 9px;
  font-weight: 600;
  color: #0F172A;
}
QToolButton[segmented="true"]:hover {
  background: #E2E8F0;
  color: #0F172A;
}
QToolButton[segmented="true"]:checked {
  background: #4F46E5;
  border: 1px solid #4F46E5;
  color: #FFFFFF;
}
QToolButton[segmented="true"]:checked:hover {
  background: #4338CA;
  border: 1px solid #4338CA;
  color: #FFFFFF;
}

QListWidget {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 10px;
}
/* NOTE: no `color:` on items — the forced light palette supplies dark text,
   and per-feature colored foregrounds (setForeground) must keep working. */
QListWidget::item {
  background: #FFFFFF;
  border-bottom: 1px solid #F1F5F9;
  padding: 5px 8px;
}
QListWidget::item:hover {
  background: #F1F5F9;
}
QListWidget::item:selected {
  background: #DDE3FA;
  border-bottom: 1px solid #C7D2FE;
}

QGroupBox { border: none; }

/* Explicit styling for states that would otherwise inherit the OS dark-mode
   palette (unreadable white-on-light text on some Macs). */
QComboBox QAbstractItemView {
  background: #FFFFFF;
  color: #0F172A;
  border: 1px solid #D1D5DB;
  selection-background-color: #EEF2FF;
  selection-color: #0F172A;
  outline: none;
}
QSpinBox {
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 8px;
  padding: 3px 6px;
  color: #0F172A;
}
QMenuBar { background: #F6F7FB; color: #0F172A; }
QMenuBar::item { background: transparent; color: #0F172A; padding: 4px 10px; }
QMenuBar::item:selected { background: #E2E8F0; color: #0F172A; border-radius: 6px; }
QMenu { background: #FFFFFF; color: #0F172A; border: 1px solid #E5E7EB; }
QMenu::item { padding: 5px 22px; color: #0F172A; }
QMenu::item:selected { background: #EEF2FF; color: #0F172A; }
QToolTip { background: #0F172A; color: #FFFFFF; border: none; padding: 5px 8px; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #F1F5F9; width: 10px; border-radius: 5px; }
QScrollBar::handle:vertical { background: #CBD5E1; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #94A3B8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #F1F5F9; height: 10px; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #CBD5E1; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QStatusBar { background: #F6F7FB; color: #475569; }
QTabWidget::pane { border: 1px solid #E5E7EB; border-radius: 8px; background: #FFFFFF; }
QTabBar::tab {
  background: #F1F5F9; color: #0F172A; padding: 6px 14px;
  border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px;
}
QTabBar::tab:selected { background: #4F46E5; color: #FFFFFF; }
QTabBar::tab:hover:!selected { background: #E2E8F0; color: #0F172A; }
"""


def _apply_light_palette(app):
    """Force a light palette so the app looks identical on dark-mode Macs.

    Without this, un-styled widget states (hovers, popups, selections) inherit
    the OS dark palette while our QSS paints light backgrounds — producing
    light-on-light (unreadable) text.
    """
    try:
        from PyQt6.QtGui import QPalette
        CR = QPalette.ColorRole
    except ImportError:
        from PyQt5.QtGui import QPalette
        CR = QPalette
    pal = QPalette()
    pal.setColor(CR.Window, QColor("#F6F7FB"))
    pal.setColor(CR.WindowText, QColor("#0F172A"))
    pal.setColor(CR.Base, QColor("#FFFFFF"))
    pal.setColor(CR.AlternateBase, QColor("#F1F5F9"))
    pal.setColor(CR.Text, QColor("#0F172A"))
    pal.setColor(CR.Button, QColor("#F1F5F9"))
    pal.setColor(CR.ButtonText, QColor("#0F172A"))
    pal.setColor(CR.Highlight, QColor("#C7D2FE"))
    pal.setColor(CR.HighlightedText, QColor("#0F172A"))
    pal.setColor(CR.ToolTipBase, QColor("#0F172A"))
    pal.setColor(CR.ToolTipText, QColor("#FFFFFF"))
    pal.setColor(CR.PlaceholderText, QColor("#94A3B8"))
    app.setPalette(pal)

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class TrackedFeature:
    name: str
    init_frame: int
    end_frame: int
    init_type: str              # "point" | "bbox"
    init_coords: list           # [x,y] or [x1,y1,x2,y2]
    color_idx: int = 0
    count: int = 1
    species_category: str = ""  # Shark / Ray / Teleost fish / Sea turtle / Other
    species: str = ""           # binomial, e.g. "Negaprion brevirostris" ("" if not to species)
    family: str = ""            # e.g. "Carcharhinidae"
    genus: str = ""             # e.g. "Negaprion"
    common: str = ""            # common name, e.g. "Lemon shark"
    confidence: str = ""        # "" | High | Medium | Low
    sex: str = ""               # "" | Male | Female | Unknown (single/first animal)
    # per-individual sexes for a group (e.g. several sharks in one interaction);
    # when non-empty, len(sexes) is the group size and drives `count`
    sexes: list = None

    def __post_init__(self):
        if self.sexes is None:
            self.sexes = []

    def id_level(self) -> str:
        """Lowest taxonomic rank this animal was actually identified to."""
        if self.species:
            return "species"
        if self.genus:
            return "genus"
        if self.family:
            return "family"
        return ""

    def id_label(self) -> str:
        """Best human-readable identification string for this animal."""
        if self.species:
            return f"{self.common} ({self.species})" if self.common else self.species
        if self.genus:
            return f"{self.genus} sp."
        if self.family:
            return self.family
        return self.species_category or ""

    def sex_summary(self) -> str:
        """Compact '2♂ 1♀ 1?' string, or '' if no per-individual sexes."""
        if not self.sexes:
            return ""
        n = {"Male": 0, "Female": 0, "Unknown": 0}
        for s in self.sexes:
            if s in n:
                n[s] += 1
        parts = []
        if n["Male"]:
            parts.append(f"{n['Male']}♂")
        if n["Female"]:
            parts.append(f"{n['Female']}♀")
        if n["Unknown"]:
            parts.append(f"{n['Unknown']}?")
        return " ".join(parts)


class FeatureStore:
    """Holds every tracked feature together with its per-frame bboxes + scene labels."""

    def __init__(self, video_path: str, fps: float, total_frames: int,
                 video_w: int, video_h: int):
        self.video_path = video_path
        self.fps = fps
        self.total_frames = total_frames
        self.video_w = video_w
        self.video_h = video_h

        self.features: List[TrackedFeature] = []
        self.feature_masks: List[Dict[int, np.ndarray]] = []
        self.feature_bboxes: List[Dict[int, Tuple[int, int, int, int]]] = []
        self._next_color = 0

        # per-frame scene labels (persist until changed).  Behavior is two
        # independent axes: movement + social/interaction.
        self.movement_per_frame: List[Optional[str]] = [None] * total_frames
        self.social_per_frame: List[Optional[str]] = [None] * total_frames
        self.habitat_per_frame: List[Optional[str]] = [None] * total_frames
        self.visibility_per_frame: List[Optional[str]] = [None] * total_frames

        # timestamped notes: list of {"frame": int, "text": str}
        self.notes: List[Dict] = []

        # a single free-text note for the whole video
        self.video_note: str = ""

        # conspecific shark encounters logged from the Social bar:
        # [{"frame": int, "males": int, "females": int, "unknown": int}]
        self.shark_log: List[Dict] = []

        # saved highlight clips: list of {"name": str, "start": int, "end": int}
        self.clips: List[Dict] = []

    def add_feature(self, feat: TrackedFeature,
                    masks: Dict[int, np.ndarray],
                    bboxes: Dict[int, Tuple[int, int, int, int]]):
        feat.color_idx = self._next_color
        self._next_color = (self._next_color + 1) % len(FEATURE_COLORS)
        self.features.append(feat)
        self.feature_masks.append(masks)
        self.feature_bboxes.append(bboxes)

    def remove_feature(self, idx: int):
        if 0 <= idx < len(self.features):
            self.features.pop(idx)
            self.feature_masks.pop(idx)
            self.feature_bboxes.pop(idx)

    def features_at(self, frame_idx: int):
        """(index, feature, mask-or-None, bbox-or-None) for every active feature."""
        out = []
        for i, feat in enumerate(self.features):
            if feat.init_frame <= frame_idx <= feat.end_frame:
                mask = self.feature_masks[i].get(frame_idx)
                bbox = self.feature_bboxes[i].get(frame_idx)
                out.append((i, feat, mask, bbox))
        return out

    def set_movement(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.movement_per_frame[frame_idx] = label

    def set_social(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.social_per_frame[frame_idx] = label

    def set_habitat(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.habitat_per_frame[frame_idx] = label

    def set_visibility(self, frame_idx: int, label: Optional[str]):
        if 0 <= frame_idx < self.total_frames:
            self.visibility_per_frame[frame_idx] = label

    def social_segment_at(self, frame: int):
        """(start, end, label) of the contiguous social-label run containing
        `frame`.  Shark-log entries apply to this whole span — e.g. a group of
        5 sharks logged during a "Brief interaction" covers the entire
        interaction, not just one moment.  Unlabeled frame → single-frame span.
        """
        frame = max(0, min(int(frame), self.total_frames - 1))
        lab = self.social_per_frame[frame]
        if lab is None:
            return frame, frame, None
        s = frame
        while s > 0 and self.social_per_frame[s - 1] == lab:
            s -= 1
        e = frame
        while e < self.total_frames - 1 and self.social_per_frame[e + 1] == lab:
            e += 1
        return s, e, lab

    def add_note(self, frame_idx: int, text: str):
        text = (text or "").strip()
        if not text:
            return
        self.notes.append({"frame": int(frame_idx), "text": text})
        self.notes.sort(key=lambda n: n["frame"])

    def remove_note(self, note: Dict):
        try:
            self.notes.remove(note)
        except ValueError:
            pass

    @staticmethod
    def _compress_timeline(labels: List[Optional[str]]):
        """Run-length encode a per-frame label list into segments."""
        if not labels:
            return []
        segs = []
        cur = labels[0]
        start = 0
        for i in range(1, len(labels)):
            if labels[i] != cur:
                segs.append({"start": start, "end": i - 1, "value": cur})
                cur = labels[i]
                start = i
        segs.append({"start": start, "end": len(labels) - 1, "value": cur})
        return segs

    def save_json(self, path: str):
        data = {
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "video_w": self.video_w,
            "video_h": self.video_h,
            "scene": {
                "movement_segments": self._compress_timeline(self.movement_per_frame),
                "social_segments": self._compress_timeline(self.social_per_frame),
                "habitat_segments": self._compress_timeline(self.habitat_per_frame),
                "visibility_segments": self._compress_timeline(self.visibility_per_frame),
            },
            "notes": [dict(n) for n in self.notes],
            "video_note": self.video_note,
            # each entry carries its derived social-segment span so analysts
            # get the full interaction window without re-deriving it
            "shark_log": [
                {**e, "start": seg[0], "end": seg[1], "social": seg[2]}
                for e in self.shark_log
                for seg in [self.social_segment_at(e["frame"])]
            ],
            "clips": [dict(c) for c in self.clips],
            "features": [],
        }
        for i, feat in enumerate(self.features):
            data["features"].append({
                "name": feat.name,
                "init_frame": feat.init_frame,
                "end_frame": feat.end_frame,
                "init_type": feat.init_type,
                "init_coords": feat.init_coords,
                "color_idx": feat.color_idx,
                "count": feat.count,
                "species_category": feat.species_category,
                "species": feat.species,
                "family": feat.family,
                "genus": feat.genus,
                "common": feat.common,
                "id_level": feat.id_level(),
                "confidence": feat.confidence,
                "sex": feat.sex,
                "sexes": list(feat.sexes or []),
                "bboxes": {str(k): list(v) for k, v in self.feature_bboxes[i].items()},
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "FeatureStore":
        """Load a previously saved annotation JSON and return a populated FeatureStore."""
        with open(path) as f:
            data = json.load(f)

        store = cls(
            video_path=data.get("video_path", ""),
            fps=float(data.get("fps", 30.0)),
            total_frames=int(data.get("total_frames", 0)),
            video_w=int(data.get("video_w", 0)),
            video_h=int(data.get("video_h", 0)),
        )

        scene = data.get("scene", {})

        def _apply(segments, setter):
            for seg in segments or []:
                for fi in range(int(seg["start"]), int(seg["end"]) + 1):
                    setter(fi, seg.get("value"))

        # New two-axis schema.  Old files had a single "behavior_segments"
        # (or legacy "environment_segments") — fold those into the movement
        # axis so no annotation is lost.
        _apply(scene.get("movement_segments",
                         scene.get("behavior_segments",
                                   scene.get("environment_segments", []))),
               store.set_movement)
        _apply(scene.get("social_segments", []), store.set_social)
        _apply(scene.get("habitat_segments", scene.get("substrate_segments", [])),
               store.set_habitat)
        # Visibility: map retired scale values onto the current 4-point scale
        # so older annotation files keep working.
        def _set_visibility_migrated(fi, value):
            store.set_visibility(fi, VISIBILITY_LEGACY_MAP.get(value, value))

        _apply(scene.get("visibility_segments", []), _set_visibility_migrated)

        for nd in data.get("notes", []):
            try:
                store.notes.append({"frame": int(nd["frame"]),
                                    "text": str(nd.get("text", ""))})
            except (KeyError, ValueError, TypeError):
                continue
        store.notes.sort(key=lambda n: n["frame"])
        store.video_note = str(data.get("video_note", "") or "")

        for ed in data.get("shark_log", []):
            try:
                store.shark_log.append({
                    "frame": int(ed["frame"]),
                    "males": int(ed.get("males", 0)),
                    "females": int(ed.get("females", 0)),
                    "unknown": int(ed.get("unknown", 0)),
                })
            except (KeyError, ValueError, TypeError):
                continue
        store.shark_log.sort(key=lambda e: e["frame"])

        for cd in data.get("clips", []):
            try:
                store.clips.append({"name": str(cd.get("name", "clip")),
                                    "start": int(cd["start"]),
                                    "end": int(cd["end"])})
            except (KeyError, ValueError, TypeError):
                continue

        def _migrate_species(fd):
            """Older files stored one free-text 'species' string that could be a
            display name, a genus ('Negaprion sp.') or a family. Split it into
            the family/genus/species fields so old work keeps its ID level."""
            raw = (fd.get("species") or "").strip()
            fam = (fd.get("family") or "").strip()
            gen = (fd.get("genus") or "").strip()
            common = (fd.get("common") or "").strip()
            if fam or gen or not raw:
                return fam, gen, raw if (fam or gen) else raw, common
            m = _SPECIES_RE.match(raw)
            if m:
                common = common or m.group(1).strip()
                gen = m.group(2)
                return GENUS_FAMILY.get(gen, ""), gen, f"{gen} {m.group(3)}", common
            if raw.endswith(" sp."):
                gen = raw[:-4].strip()
                return GENUS_FAMILY.get(gen, ""), gen, "", ""
            if raw.endswith("idae"):
                return raw, "", "", ""
            parts = raw.split()
            if len(parts) == 2 and parts[0][:1].isupper() and parts[1].islower():
                return GENUS_FAMILY.get(parts[0], ""), parts[0], raw, common
            return "", "", raw, common

        for fd in data.get("features", []):
            _fam, _gen, _sp, _common = _migrate_species(fd)
            feat = TrackedFeature(
                name=fd.get("name", "feature"),
                init_frame=int(fd.get("init_frame", 0)),
                end_frame=int(fd.get("end_frame", 0)),
                init_type=fd.get("init_type", "point"),
                init_coords=fd.get("init_coords", [0, 0]),
                color_idx=int(fd.get("color_idx", 0)),
                count=int(fd.get("count", 1)),
                species_category=fd.get("species_category", fd.get("feature_type", "")),
                species=_sp,
                family=_fam,
                genus=_gen,
                common=_common,
                confidence=fd.get("confidence", ""),
                sex=fd.get("sex", ""),
                sexes=list(fd.get("sexes", []) or []),
            )
            bboxes_raw = fd.get("bboxes", {})
            bboxes: Dict[int, Tuple[int, int, int, int]] = {
                int(k): tuple(v) for k, v in bboxes_raw.items()  # type: ignore[misc]
            }
            store.features.append(feat)
            store.feature_masks.append({})
            store.feature_bboxes.append(bboxes)
            store._next_color = (feat.color_idx + 1) % len(FEATURE_COLORS)

        return store


# --------------------------------------------------------------------------
# VideoLabel — displays frames; click/drag signals kept for possible future use
# --------------------------------------------------------------------------

class FlowLayout(QLayout):
    """A layout that wraps its widgets onto new rows when width runs out.

    The label bars (Movement/Social/Habitat/Visibility) used a QHBoxLayout,
    whose minimum width is the sum of every button — that forced the whole
    window wider than a 13" laptop screen and pushed the control panel off
    the display. Wrapping lets a bar shrink to one button wide and grow
    taller instead, so the window fits any screen.
    """

    def __init__(self, parent=None, margin=0, spacing=5):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def __del__(self):
        while self.count():
            self.takeAt(0)

    def addItem(self, item):
        self._items.append(item)

    def insertWidget(self, index, widget):
        self.addWidget(widget)
        item = self._items.pop()
        self._items.insert(max(0, min(index, len(self._items))), item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        x, y = rect.x() + m.left(), rect.y() + m.top()
        right = rect.right() - m.right()
        line_height = 0
        space = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + space
            if next_x - space > right and line_height > 0:
                x = rect.x() + m.left()
                y = y + line_height + space
                next_x = x + hint.width() + space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()


class FlowBar(QWidget):
    """Container for a FlowLayout that reports the height its content actually
    needs at the width it was given.

    Qt does not reliably push a parent's width down into a child's
    heightForWidth, so a wrapping bar nested in a column would otherwise be
    laid out at its minimum width and claim far more height than it needs.
    Recomputing on resize keeps the bars exactly as tall as their rows.
    """

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        lay = self.layout()
        if lay is None:
            return
        h = lay.heightForWidth(self.width())
        if h > 0 and h != self.minimumHeight():
            self.setMinimumHeight(h)
            self.setMaximumHeight(h)


class VideoLabel(QLabel):
    pointClicked = pyqtSignal(object)           # QPoint — click on empty area
    bboxDrawn = pyqtSignal(object, object)      # two-click: start,end (label coords)
    bboxEdited = pyqtSignal(int, int, int, int)  # edited box (label coords)
    zoomRequested = pyqtSignal(int, object)      # wheel delta, anchor QPoint
    firstCornerPlaced = pyqtSignal()             # first of the two draw-clicks

    HANDLE = 9  # corner-handle hit radius / draw size (px)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._bbox_mode = False
        # two-click drawing state
        self._first_corner = None       # QPoint (label coords) or None
        self._cursor_pos = None
        # corner-edit state
        self._edit_rect = None          # [x1,y1,x2,y2] label coords of selected box
        self._active_handle = None      # 0..3 corner index, "move", or None
        self._drag_origin = None

    # -- modes ------------------------------------------------------------
    def set_bbox_mode(self, on: bool):
        self._bbox_mode = on
        self._first_corner = None
        cross = Qt.CursorShape.CrossCursor if _QT6 else Qt.CrossCursor
        arrow = Qt.CursorShape.ArrowCursor if _QT6 else Qt.ArrowCursor
        self.setCursor(cross if on else arrow)
        self.update()

    def set_edit_rect(self, rect):
        """Set the selected feature's box (label coords) for corner editing."""
        self._edit_rect = list(rect) if rect else None
        self.update()

    def set_frame_pixmap(self, pix):
        self.setPixmap(pix)

    @staticmethod
    def _ev_pos(ev):
        if _QT6:
            return ev.position().toPoint()
        return ev.pos()

    # -- hit testing ------------------------------------------------------
    def _handle_at(self, pos):
        if not self._edit_rect:
            return None
        x1, y1, x2, y2 = self._edit_rect
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i, (cx, cy) in enumerate(corners):
            if abs(pos.x() - cx) <= self.HANDLE and abs(pos.y() - cy) <= self.HANDLE:
                return i
        if min(x1, x2) <= pos.x() <= max(x1, x2) and \
           min(y1, y2) <= pos.y() <= max(y1, y2):
            return "move"
        return None

    # -- mouse ------------------------------------------------------------
    def mousePressEvent(self, ev):
        btn = Qt.MouseButton.LeftButton if _QT6 else Qt.LeftButton
        rbtn = Qt.MouseButton.RightButton if _QT6 else Qt.RightButton
        pos = self._ev_pos(ev)

        if ev.button() == rbtn:
            # right-click cancels an in-progress draw
            self._first_corner = None
            self.update()
            return

        if ev.button() != btn:
            return

        if self._bbox_mode:
            if self._first_corner is None:
                self._first_corner = pos
                self._cursor_pos = pos
                self.firstCornerPlaced.emit()
            else:
                self.bboxDrawn.emit(self._first_corner, pos)
                self._first_corner = None
            self.update()
            return

        # not drawing → maybe editing a selected box, else a plain click
        h = self._handle_at(pos)
        if h is not None:
            self._active_handle = h
            self._drag_origin = pos
        else:
            self.pointClicked.emit(pos)

    def mouseMoveEvent(self, ev):
        pos = self._ev_pos(ev)
        if self._bbox_mode and self._first_corner is not None:
            self._cursor_pos = pos
            self.update()
            return
        if self._active_handle is not None and self._edit_rect is not None:
            self._drag_handle(pos)
            self.update()
            return
        # hover cursor feedback over edit handles
        if not self._bbox_mode and self._edit_rect is not None:
            h = self._handle_at(pos)
            if h == "move":
                self.setCursor(Qt.CursorShape.SizeAllCursor if _QT6 else Qt.SizeAllCursor)
            elif h is not None:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor if _QT6 else Qt.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor if _QT6 else Qt.ArrowCursor)

    def _drag_handle(self, pos):
        x1, y1, x2, y2 = self._edit_rect
        if self._active_handle == "move":
            dx = pos.x() - self._drag_origin.x()
            dy = pos.y() - self._drag_origin.y()
            self._edit_rect = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]
            self._drag_origin = pos
        else:
            # move only the dragged corner
            corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y1]]
            i = self._active_handle
            if i == 0:
                self._edit_rect = [pos.x(), pos.y(), x2, y2]
            elif i == 1:
                self._edit_rect = [x1, pos.y(), pos.x(), y2]
            elif i == 2:
                self._edit_rect = [x1, y1, pos.x(), pos.y()]
            elif i == 3:
                self._edit_rect = [pos.x(), y1, x2, pos.y()]

    def mouseReleaseEvent(self, ev):
        if self._active_handle is not None and self._edit_rect is not None:
            x1, y1, x2, y2 = self._edit_rect
            self.bboxEdited.emit(int(min(x1, x2)), int(min(y1, y2)),
                                 int(max(x1, x2)), int(max(y1, y2)))
            self._active_handle = None
            self._drag_origin = None

    def wheelEvent(self, ev):
        mods = ev.modifiers()
        ctrl = Qt.KeyboardModifier.ControlModifier if _QT6 else Qt.ControlModifier
        meta = Qt.KeyboardModifier.MetaModifier if _QT6 else Qt.MetaModifier
        if mods & ctrl or mods & meta:
            delta = ev.angleDelta().y()
            pos = ev.position().toPoint() if _QT6 else ev.pos()
            self.zoomRequested.emit(delta, pos)
            ev.accept()
        else:
            super().wheelEvent(ev)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        if _QT6:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen_solid = Qt.PenStyle.SolidLine if _QT6 else Qt.SolidLine
        no_pen = Qt.PenStyle.NoPen if _QT6 else Qt.NoPen

        # two-click preview
        if self._bbox_mode and self._first_corner is not None and self._cursor_pos:
            p.setPen(QPen(QColor(79, 70, 229), 2, pen_solid))
            x1, y1 = self._first_corner.x(), self._first_corner.y()
            x2, y2 = self._cursor_pos.x(), self._cursor_pos.y()
            p.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            p.setPen(no_pen)
            p.setBrush(QColor(79, 70, 229))
            p.drawEllipse(x1 - 4, y1 - 4, 8, 8)

        # edit handles on the selected box
        if self._edit_rect is not None and not self._bbox_mode:
            x1, y1, x2, y2 = self._edit_rect
            p.setPen(QPen(QColor(16, 185, 129), 2, pen_solid))
            p.setBrush(Qt.BrushStyle.NoBrush if _QT6 else Qt.NoBrush)
            p.drawRect(int(min(x1, x2)), int(min(y1, y2)),
                       int(abs(x2 - x1)), int(abs(y2 - y1)))
            p.setBrush(QColor(16, 185, 129))
            p.setPen(QPen(QColor(255, 255, 255), 1, pen_solid))
            hs = self.HANDLE
            for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                p.drawRect(int(cx - hs / 2), int(cy - hs / 2), hs, hs)


# --------------------------------------------------------------------------
# Bottom Timeline Widget (habitat + features + timestamp scrubber)
# --------------------------------------------------------------------------

class AnnotationTimeline(QWidget):
    """
    Bottom timeline widget:
      - Habitat/Environment segments over time
      - Feature presence segments over time (color-coded)
      - Scrubber row with big knob + timestamp labels
    Click/drag anywhere to seek.
    """
    frameSelected = pyqtSignal(int)

    def __init__(self, store: FeatureStore, fps: float, total_frames: int, parent=None):
        super().__init__(parent)
        self.store = store
        self.fps = float(fps) if fps else 30.0
        self.total_frames = max(1, int(total_frames))
        self.current_frame = 0
        self._dragging = False
        # highlight-clip in/out marks (set by MainWindow)
        self.clip_in = None
        self.clip_out = None

        # Size policy compatibility
        try:
            exp = QSizePolicy.Policy.Expanding
            fixed = QSizePolicy.Policy.Fixed
        except Exception:
            exp = QSizePolicy.Expanding
            fixed = QSizePolicy.Fixed

        self.setSizePolicy(exp, fixed)
        self.setMinimumHeight(142)
        self.setMaximumHeight(150)
        self.setMouseTracking(True)

        # Colors for movement + social segments — cycle FEATURE_COLORS palette
        self._movement_colors = {
            opt: QColor(*FEATURE_COLORS[i % len(FEATURE_COLORS)])
            for i, opt in enumerate(MOVEMENT_OPTIONS)
        }
        self._movement_colors[None] = QColor(203, 213, 225)
        self._social_colors = {
            opt: QColor(*FEATURE_COLORS[(i + 3) % len(FEATURE_COLORS)])
            for i, opt in enumerate(SOCIAL_OPTIONS)
        }
        # retired labels keep a colour so older annotations still render
        for j, opt in enumerate(sorted(RETIRED_SOCIAL)):
            self._social_colors.setdefault(
                opt, QColor(*FEATURE_COLORS[(len(SOCIAL_OPTIONS) + j + 3)
                                            % len(FEATURE_COLORS)]))
        self._social_colors[None] = QColor(203, 213, 225)

        # Colors for habitat segments — known habitats get fixed colors, any
        # user-added ones fall back to the palette by index.
        _known_hab = {
            "Mangrove": QColor(16, 185, 129),
            "Rocky reef": QColor(14, 165, 233),
            "Sandy bottom": QColor(245, 158, 11),
            "Gravel": QColor(156, 163, 175),
            "Mud": QColor(120, 100, 80),
        }
        self._habitat_colors = {None: QColor(203, 213, 225)}
        for i, h in enumerate(HABITAT_OPTIONS):
            self._habitat_colors[h] = _known_hab.get(
                h, QColor(*FEATURE_COLORS[i % len(FEATURE_COLORS)]))

        # Colors for the 5-point visibility scale (red → green)
        self._visibility_colors = {
            "Very bad": QColor(190, 18, 60),
            "Bad": QColor(239, 68, 68),
            "Fine": QColor(245, 158, 11),
            "Good": QColor(34, 197, 94),
            "Very good": QColor(21, 128, 61),
            None: QColor(203, 213, 225),
        }

    def set_current_frame(self, idx: int):
        self.current_frame = max(0, min(int(idx), self.total_frames - 1))
        self.update()

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        s = int(seconds + 0.5)
        h = s // 3600
        m = (s % 3600) // 60
        ss = s % 60
        if h > 0:
            return f"{h}:{m:02d}:{ss:02d}"
        return f"{m}:{ss:02d}"

    def _bar_geometry(self):
        label_w = 150
        pad_l = 14
        pad_r = 14
        x0 = pad_l + label_w
        x1 = self.width() - pad_r
        w = max(1, x1 - x0)
        return label_w, x0, x1, w

    def _frame_to_x(self, frame_idx: int, x0: int, w: int) -> int:
        if self.total_frames <= 1:
            return x0
        t = frame_idx / (self.total_frames - 1)
        return int(x0 + t * w)

    def _x_to_frame(self, x: int, x0: int, w: int) -> int:
        x = max(x0, min(x0 + w, x))
        if w <= 0 or self.total_frames <= 1:
            return 0
        t = (x - x0) / w
        return int(round(t * (self.total_frames - 1)))

    def mousePressEvent(self, ev):
        btn = Qt.MouseButton.LeftButton if _QT6 else Qt.LeftButton
        if ev.button() != btn:
            return
        _, x0, _, w = self._bar_geometry()
        pos = ev.position().toPoint() if _QT6 else ev.pos()
        if pos.x() < x0 - 10:
            return
        self._dragging = True
        f = self._x_to_frame(pos.x(), x0, w)
        self.set_current_frame(f)
        self.frameSelected.emit(f)

    def mouseMoveEvent(self, ev):
        if not self._dragging:
            return
        _, x0, _, w = self._bar_geometry()
        pos = ev.position().toPoint() if _QT6 else ev.pos()
        f = self._x_to_frame(pos.x(), x0, w)
        self.set_current_frame(f)
        self.frameSelected.emit(f)

    def mouseReleaseEvent(self, ev):
        self._dragging = False

    def paintEvent(self, ev):
        p = QPainter(self)
        if _QT6:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        else:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setRenderHint(QPainter.TextAntialiasing, True)

        label_w, x0, x1, w = self._bar_geometry()
        top = 3
        row_h = 17
        gap = 3

        movement_y = top
        social_y = movement_y + row_h + gap
        habitat_y = social_y + row_h + gap
        visibility_y = habitat_y + row_h + gap
        animals_y = visibility_y + row_h + gap
        scrub_y = animals_y + row_h + gap

        # Background
        p.fillRect(self.rect(), QColor(246, 247, 251))

        def draw_left_label(text: str, y: int):
            p.setPen(QColor(15, 23, 42))
            f = p.font()
            f.setPointSize(11)
            f.setBold(True)
            p.setFont(f)
            p.drawText(14, y + row_h - 4, text)

        def draw_bar_outline(y: int):
            p.setPen(QPen(QColor(148, 163, 184), 1))
            p.setBrush(QColor(255, 255, 255))
            p.drawRoundedRect(x0, y, w, row_h, 6, 6)

        def draw_segment_row(row_y, labels, color_dict):
            draw_bar_outline(row_y)
            the_segs = FeatureStore._compress_timeline(
                labels if labels is not None else [None] * self.total_frames
            )
            for seg in the_segs:
                v = seg.get("value", None)
                c = color_dict.get(v, QColor(203, 213, 225))
                s = int(seg["start"])
                e = int(seg["end"])
                x_s = int(x0 + (s / max(1, self.total_frames)) * w)
                x_e = int(x0 + ((e + 1) / max(1, self.total_frames)) * w)
                if x_e <= x_s:
                    continue
                p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
                p.setBrush(QColor(c.red(), c.green(), c.blue(), 70))
                p.drawRoundedRect(x_s, row_y + 1, max(1, x_e - x_s), row_h - 2, 6, 6)
                lbl = v if v is not None else "Unknown"
                if (x_e - x_s) > 90:
                    p.setPen(QColor(15, 23, 42))
                    ff = p.font()
                    ff.setBold(False)
                    ff.setPointSize(11)
                    p.setFont(ff)
                    p.drawText(x_s + 8, row_y + row_h - 6, lbl)

        # --- Movement row
        draw_left_label("Movement", movement_y)
        movement_labels = self.store.movement_per_frame if self.store is not None else None
        draw_segment_row(movement_y, movement_labels, self._movement_colors)

        # --- Social row
        draw_left_label("Social", social_y)
        social_labels = self.store.social_per_frame if self.store is not None else None
        draw_segment_row(social_y, social_labels, self._social_colors)

        # shark-log bands on the social row: each entry covers its whole
        # social-interaction segment (dark underline band + ◆ + count)
        if self.store is not None and getattr(self.store, "shark_log", None):
            for e in self.store.shark_log:
                seg_s, seg_e, _lab = self.store.social_segment_at(e["frame"])
                bx_s = int(x0 + (seg_s / max(1, self.total_frames)) * w)
                bx_e = int(x0 + ((seg_e + 1) / max(1, self.total_frames)) * w)
                band_y = social_y + row_h - 5
                p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
                p.setBrush(QColor(15, 23, 42, 210))
                p.drawRoundedRect(bx_s, band_y, max(3, bx_e - bx_s), 4, 2, 2)
                # diamond at the exact log frame
                ex = int(x0 + (e["frame"] / max(1, self.total_frames)) * w)
                cy = band_y + 2
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.setBrush(QColor(15, 23, 42))
                pts = [(ex, cy - 5), (ex + 5, cy), (ex, cy + 5), (ex - 5, cy)]
                p.drawPolygon(QPolygon([QPoint(a, b) for a, b in pts]))
                # total count right after the band (or inside its right end)
                total = (e.get("males", 0) + e.get("females", 0)
                         + e.get("unknown", 0))
                ff = p.font(); ff.setPointSize(9); ff.setBold(True); p.setFont(ff)
                p.setPen(QColor(15, 23, 42))
                tx = min(bx_e + 4, x0 + w - 14)
                p.drawText(tx, band_y + 4, str(total))

        # --- Habitat row
        draw_left_label("Habitat", habitat_y)
        habitat_labels = self.store.habitat_per_frame if self.store is not None else None
        draw_segment_row(habitat_y, habitat_labels, self._habitat_colors)

        # --- Visibility row
        draw_left_label("Visibility", visibility_y)
        visibility_labels = self.store.visibility_per_frame if self.store is not None else None
        draw_segment_row(visibility_y, visibility_labels, self._visibility_colors)

        # --- Other animals row
        draw_left_label("Features:", animals_y)
        draw_bar_outline(animals_y)

        intervals = []
        if self.store is not None:
            for feat in self.store.features:
                intervals.append((feat.init_frame, feat.end_frame, feat))
        intervals.sort(key=lambda t: (t[0], t[1]))

        # Greedy lane assignment (compact)
        max_lanes = 3
        lane_ends = [-1] * max_lanes
        assigned = []
        for s, e, feat in intervals:
            lane = None
            for li in range(max_lanes):
                if s > lane_ends[li]:
                    lane = li
                    lane_ends[li] = e
                    break
            if lane is None:
                lane = 0
            assigned.append((s, e, feat, lane))

        lane_h = max(6, (row_h - 4) // max_lanes)

        for s, e, feat, lane in assigned:
            x_s = int(x0 + (s / max(1, self.total_frames)) * w)
            x_e = int(x0 + ((e + 1) / max(1, self.total_frames)) * w)
            if x_e <= x_s:
                continue

            c_rgb = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)]
            c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])

            y = animals_y + 2 + lane * lane_h
            h = lane_h - 2

            p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 170))
            p.drawRoundedRect(x_s, y, max(2, x_e - x_s), h, 4, 4)

            # Name tag above row
            if (x_e - x_s) > 65:
                p.setPen(QColor(15, 23, 42))
                f = p.font()
                f.setPointSize(10)
                f.setBold(False)
                p.setFont(f)
                # clamp to not go out of bounds
                tx = min(max(x_s + 2, x0), x0 + w - 60)
                p.drawText(tx, animals_y - 3, feat.name)

        # --- Scrubber row (timestamps)
        draw_bar_outline(scrub_y)

        # mid guide line
        p.setPen(QPen(QColor(148, 163, 184), 2))
        mid_y = scrub_y + row_h // 2
        p.drawLine(x0 + 8, mid_y, x0 + w - 8, mid_y)

        # saved highlight clips (amber bands) + current in/out (indigo band)
        def _fx(frame):
            return int(x0 + (frame / max(1, self.total_frames)) * w)

        if self.store is not None:
            p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
            for c in getattr(self.store, "clips", []):
                xs, xe = _fx(c["start"]), _fx(c["end"] + 1)
                p.setBrush(QColor(245, 158, 11, 90))
                p.drawRoundedRect(xs, scrub_y + 2, max(2, xe - xs), row_h - 4, 4, 4)

            # comment markers — small rose flags so you can spot "come back
            # here" frames at a glance
            p.setBrush(QColor(244, 63, 94))
            p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
            for note in getattr(self.store, "notes", []):
                nx = _fx(int(note.get("frame", 0)))
                p.drawEllipse(nx - 3, scrub_y - 4, 6, 6)
                p.drawRect(nx - 1, scrub_y - 1, 2, row_h // 2)

        if self.clip_in is not None or self.clip_out is not None:
            a = self.clip_in if self.clip_in is not None else self.clip_out
            b = self.clip_out if self.clip_out is not None else self.clip_in
            a, b = min(a, b), max(a, b)
            xs, xe = _fx(a), _fx(b + 1)
            p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
            p.setBrush(QColor(79, 70, 229, 80))
            p.drawRoundedRect(xs, scrub_y + 2, max(2, xe - xs), row_h - 4, 4, 4)
            p.setPen(QPen(QColor(79, 70, 229), 2))
            for m in (self.clip_in, self.clip_out):
                if m is not None:
                    mx = _fx(m)
                    p.drawLine(mx, scrub_y, mx, scrub_y + row_h)

        # Cursor line + knob
        cx = self._frame_to_x(self.current_frame, x0, w)
        p.setPen(QPen(QColor(15, 23, 42), 2))
        p.drawLine(cx, movement_y, cx, scrub_y + row_h)

        p.setBrush(QColor(15, 23, 42))
        p.setPen(Qt.PenStyle.NoPen if _QT6 else Qt.NoPen)
        p.drawEllipse(cx - 9, mid_y - 9, 18, 18)

        # Time labels
        p.setPen(QColor(100, 116, 139))
        f = p.font()
        f.setPointSize(10)
        f.setBold(False)
        p.setFont(f)

        start_t = self._format_time(0)
        end_t = self._format_time((self.total_frames - 1) / max(1e-6, self.fps))
        cur_t = self._format_time(self.current_frame / max(1e-6, self.fps))

        p.drawText(x0, scrub_y + row_h + 16, start_t)
        end_w = p.fontMetrics().horizontalAdvance(end_t)
        p.drawText(x0 + w - end_w, scrub_y + row_h + 16, end_t)

        # current time beside the knob, inside the scrub row (bold)
        p.setPen(QColor(15, 23, 42))
        fb = p.font(); fb.setBold(True); p.setFont(fb)
        cur_w = p.fontMetrics().horizontalAdvance(cur_t)
        ty = mid_y + p.fontMetrics().ascent() // 2 - 1
        if cx + 14 + cur_w <= x0 + w - 4:
            p.drawText(cx + 14, ty, cur_t)      # right of knob
        else:
            p.drawText(cx - 14 - cur_w, ty, cur_t)  # left of knob near the end
        fb.setBold(False); p.setFont(fb)


# --------------------------------------------------------------------------
# Startup chooser — pick a single video or a whole folder, with progress
# --------------------------------------------------------------------------

RECENT_JSON = os.path.join(CONFIG_DIR, "recent.json")
SETTINGS_JSON = os.path.join(CONFIG_DIR, "settings.json")


def load_settings():
    """UI preferences that should survive quitting the app."""
    try:
        with open(SETTINGS_JSON) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(d):
    try:
        _ensure_config_dir()
        with open(SETTINGS_JSON, "w") as f:
            json.dump(d, f, indent=2)
    except OSError:
        pass


def load_recent():
    try:
        with open(RECENT_JSON) as f:
            data = json.load(f)
        return [d for d in data.get("folders", []) if os.path.isdir(d)][:5]
    except (OSError, ValueError):
        return []


def remember_recent(folder):
    if not folder or not os.path.isdir(folder):
        return
    items = [folder] + [f for f in load_recent() if f != folder]
    try:
        _ensure_config_dir()
        with open(RECENT_JSON, "w") as f:
            json.dump({"folders": items[:5]}, f, indent=2)
    except OSError:
        pass


class StartupDialog(QDialog):
    """Asks what to work on instead of dumping the user straight into a file
    picker. Folder mode scans for existing annotations so work resumes at the
    first unannotated clip rather than replaying finished ones."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CTAG Annotator")
        self.result_choice = None      # ("video"|"frames"|"folder", path)
        self.setMinimumWidth(520)

        v = QVBoxLayout(self)
        v.setContentsMargins(22, 20, 22, 18)
        v.setSpacing(12)

        title = QLabel("What would you like to annotate?")
        title.setObjectName("Title")
        v.addWidget(title)

        sub = QLabel("Open a whole deployment folder to work through it clip by "
                     "clip — already-annotated videos are detected so you pick "
                     "up where you left off.")
        sub.setObjectName("Subtle")
        sub.setWordWrap(True)
        v.addWidget(sub)

        folder_btn = QPushButton("Open a folder of videos…")
        folder_btn.clicked.connect(self._pick_folder)
        v.addWidget(folder_btn)

        row = QHBoxLayout()
        row.setSpacing(8)
        one_btn = QPushButton("Open a single video…")
        one_btn.setObjectName("Secondary")
        one_btn.clicked.connect(self._pick_video)
        frames_btn = QPushButton("Open a frames folder…")
        frames_btn.setObjectName("Secondary")
        frames_btn.clicked.connect(self._pick_frames)
        row.addWidget(one_btn)
        row.addWidget(frames_btn)
        v.addLayout(row)

        recents = load_recent()
        if recents:
            lbl = QLabel("Recent folders")
            lbl.setObjectName("BarTitle")
            v.addWidget(lbl)
            self.recent_list = QListWidget()
            self.recent_list.setMaximumHeight(112)
            for folder in recents:
                vids = list_videos_in_folder(folder)
                done = sum(1 for p in vids if is_annotated(p))
                it = QListWidgetItem(
                    f"{os.path.basename(folder.rstrip('/'))}   —   "
                    f"{done}/{len(vids)} annotated")
                it.setToolTip(folder)
                it.setData(Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole, folder)
                self.recent_list.addItem(it)
            self.recent_list.itemDoubleClicked.connect(self._use_recent)
            v.addWidget(self.recent_list)

        quit_row = QHBoxLayout()
        quit_row.addStretch(1)
        q = QPushButton("Quit")
        q.setObjectName("Secondary")
        q.clicked.connect(self.reject)
        quit_row.addWidget(q)
        v.addLayout(quit_row)

    def _use_recent(self, item):
        role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        self._accept_folder(item.data(role))

    def _pick_video(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Video (*.mp4 *.MP4 *.mov *.MOV *.avi *.mkv)")
        if p:
            self.result_choice = ("video", p)
            self.accept()

    def _pick_frames(self):
        d = QFileDialog.getExistingDirectory(self, "Select a folder of JPEG frames")
        if d:
            self.result_choice = ("frames", d)
            self.accept()

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select a folder of videos")
        if d:
            self._accept_folder(d)

    def _accept_folder(self, d):
        vids = list_videos_in_folder(d)
        if not vids:
            QMessageBox.information(
                self, "No videos found",
                f"No video files in:\n{d}\n\n"
                "If this folder holds extracted JPEG frames, use "
                "'Open a frames folder…' instead.")
            return
        done = [p for p in vids if is_annotated(p)]
        todo = [p for p in vids if not is_annotated(p)]
        if done and todo:
            yes = QMessageBox.StandardButton.Yes if _QT6 else QMessageBox.Yes
            no = QMessageBox.StandardButton.No if _QT6 else QMessageBox.No
            ret = QMessageBox.question(
                self, "Resume this folder?",
                f"{len(vids)} videos found — {len(done)} already annotated, "
                f"{len(todo)} still to do.\n\n"
                f"Start at the first unannotated clip "
                f"({os.path.basename(todo[0])})?\n\n"
                "Choosing No starts at the beginning instead. Either way every "
                "clip stays available in the dropdown.",
                yes | no)
            start = todo[0] if ret == yes else vids[0]
        elif done and not todo:
            QMessageBox.information(
                self, "All done",
                f"All {len(vids)} videos in this folder are already annotated. "
                "Opening the first one.")
            start = vids[0]
        else:
            start = vids[0]
        remember_recent(d)
        self.result_choice = ("folder", (d, vids, start))
        self.accept()


# --------------------------------------------------------------------------
# Taxonomy picker — record an ID at whatever rank the annotator can manage
# --------------------------------------------------------------------------

class TaxonomyPicker(QWidget):
    """Family / Genus / Species pickers plus a single 'find' box.

    Two ways to work, so nobody is forced through steps they don't need:
      • Know the animal → type it in "Find" (or straight into Species) and the
        higher ranks back-fill automatically.
      • Only know it to family/genus → fill just that box and stop; the ID
        level is recorded honestly as family or genus.
    Choosing a family narrows the genus and species lists for browsing.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating = False
        self._category = ""
        self._find_map = {}
        self._common = ""
        self._browsable = []

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # --- quick find across every rank at once
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find species / genus / family…")
        self.find_edit.setClearButtonEnabled(True)
        self._find_completer = QCompleter([], self)
        self._set_contains(self._find_completer)
        self._find_completer.setMaxVisibleItems(14)
        self.find_edit.setCompleter(self._find_completer)
        self.find_edit.textEdited.connect(self._refresh_find_model)
        self._find_completer.activated[str].connect(self._on_find_activated)
        find_row = QHBoxLayout()
        find_row.setSpacing(6)
        fl = QLabel("Find:")
        fl.setMinimumWidth(52)
        find_row.addWidget(fl)
        find_row.addWidget(self.find_edit, stretch=1)
        lay.addLayout(find_row)

        # --- the three rank boxes
        self.family_combo = self._mk_combo("Family")
        self.genus_combo = self._mk_combo("Genus")
        self.species_combo = self._mk_combo("Species")
        for label, combo, adder in (("Family:", self.family_combo, None),
                                    ("Genus:", self.genus_combo, None),
                                    ("Species:", self.species_combo, self._add_species)):
            row = QHBoxLayout()
            row.setSpacing(6)
            lb = QLabel(label)
            lb.setMinimumWidth(52)
            row.addWidget(lb)
            row.addWidget(combo, stretch=1)
            if adder is not None:
                b = QToolButton()
                b.setText("＋")
                b.setProperty("segmented", True)
                b.setToolTip("Add a new species to the taxonomy")
                b.clicked.connect(adder)
                row.addWidget(b)
            lay.addLayout(row)

        self.family_combo.currentTextChanged.connect(self._on_family)
        self.genus_combo.currentTextChanged.connect(self._on_genus)
        self.species_combo.currentTextChanged.connect(self._on_species)

        # --- honest read-out of how far the ID got
        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.level_lbl = QLabel("Not identified")
        self.level_lbl.setObjectName("Subtle")
        clear_btn = QToolButton()
        clear_btn.setText("Clear")
        clear_btn.setProperty("segmented", True)
        clear_btn.clicked.connect(self.clear)
        bottom.addWidget(self.level_lbl, stretch=1)
        bottom.addWidget(clear_btn)
        lay.addLayout(bottom)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _set_contains(completer):
        try:
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        except AttributeError:
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setCompletionMode(QCompleter.PopupCompletion)

    def _mk_combo(self, placeholder):
        """An editable combo that is also fully browsable: click the caret (or
        the empty field) for the whole scrollable list, or type to filter it
        live — the same feel as a guessing-game country picker."""
        c = QComboBox()
        c.setEditable(True)
        c.setInsertPolicy(QComboBox.InsertPolicy.NoInsert if _QT6
                          else QComboBox.NoInsert)
        try:
            c.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        except AttributeError:
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(10)
        c.setMaxVisibleItems(16)          # long, scrollable popup
        le = c.lineEdit()
        le.setPlaceholderText(placeholder)
        le.setClearButtonEnabled(True)
        self._set_contains(c.completer())
        # clicking into an empty box pops the full list, so browsing is
        # discoverable without knowing to hit the caret
        le.installEventFilter(self)
        self._browsable.append(c)
        return c

    def eventFilter(self, obj, ev):
        try:
            is_press = ev.type() == ev.Type.MouseButtonPress
        except AttributeError:
            is_press = ev.type() == ev.MouseButtonPress
        if is_press:
            for c in self._browsable:
                if c.lineEdit() is obj and not c.lineEdit().text().strip():
                    QTimer.singleShot(0, c.showPopup)
                    break
        return super().eventFilter(obj, ev)

    @staticmethod
    def _fill(combo, items, keep=""):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("")
        combo.addItems(items)
        combo.setCurrentIndex(0)
        # tell the annotator how much is in here / how far it narrowed
        base = combo.lineEdit().placeholderText().split("  (")[0]
        combo.lineEdit().setPlaceholderText(
            f"{base}  ({len(items)})" if items else base)
        if keep:
            i = combo.findText(keep)
            if i >= 0:
                combo.setCurrentIndex(i)
            else:
                combo.setEditText(keep)
        # show the START of a long name rather than its scrolled-to-end tail
        if combo.lineEdit() is not None:
            combo.lineEdit().setCursorPosition(0)
        combo.blockSignals(False)

    @staticmethod
    def _species_display(row):
        return f"{row['common']} ({row['species']})" if row.get("common") else row["species"]

    # ------------------------------------------------------------- find box
    def _refresh_find_model(self, text):
        hits = TAXONOMY.search(text, self._category or None)
        labels, self._find_map = [], {}
        for h in hits:
            tag = {"species": "species", "genus": "genus", "family": "family"}[h["kind"]]
            label = f"{h['label']}   · {tag}"
            labels.append(label)
            self._find_map[label] = h
        try:
            self._find_completer.model().setStringList(labels)
        except AttributeError:
            from PyQt6.QtCore import QStringListModel
            self._find_completer.setModel(QStringListModel(labels))

    def _on_find_activated(self, label):
        h = self._find_map.get(label)
        if not h:
            return
        row = h["row"]
        if h["kind"] == "species":
            self.set_value(row.get("family", ""), row.get("genus", ""),
                           row.get("species", ""), row.get("common", ""))
        elif h["kind"] == "genus":
            self.set_value(row.get("family", ""), row.get("genus", ""), "", "")
        else:
            self.set_value(row.get("family", ""), "", "", "")
        self.find_edit.clear()
        self.changed.emit()

    # ------------------------------------------------------- cascade logic
    def _on_family(self, fam):
        if self._updating:
            return
        self._updating = True
        fam = family_from_display(fam)
        genus = self.genus_combo.currentText().strip()
        if genus and TAXONOMY.family_of_genus(genus) != fam:
            genus = ""
        self._fill(self.genus_combo, TAXONOMY.genera(self._category or None, fam or None), genus)
        self._refill_species(fam, genus)
        self._updating = False
        self._sync_level()
        self.changed.emit()

    def _on_genus(self, gen):
        if self._updating:
            return
        self._updating = True
        gen = gen.strip()
        fam = (TAXONOMY.family_of_genus(gen)
               or family_from_display(self.family_combo.currentText()))
        if fam and fam != family_from_display(self.family_combo.currentText()):
            self._fill(self.family_combo,
                       [family_display(f) for f in TAXONOMY.families(self._category or None)],
                       family_display(fam))
        self._refill_species(fam, gen)
        self._updating = False
        self._sync_level()
        self.changed.emit()

    def _on_species(self, disp):
        if self._updating:
            return
        self._updating = True
        disp = disp.strip()
        row = None
        for r in TAXONOMY.species_rows(self._category or None):
            if self._species_display(r) == disp or r["species"] == disp:
                row = r
                break
        if row:
            self._common = row.get("common", "")
            fam, gen = row.get("family", ""), row.get("genus", "")
            if fam:
                self._fill(self.family_combo,
                           [family_display(f) for f in TAXONOMY.families(self._category or None)],
                           family_display(fam))
            if gen:
                self._fill(self.genus_combo,
                           TAXONOMY.genera(self._category or None, fam or None), gen)
        else:
            self._common = ""
        self._updating = False
        self._sync_level()
        self.changed.emit()

    def _refill_species(self, family, genus):
        rows = TAXONOMY.species_rows(self._category or None,
                                     family or None, genus or None)
        cur = self.species_combo.currentText().strip()
        labels = [self._species_display(r) for r in rows]
        self._fill(self.species_combo, labels, cur if cur in labels else "")

    # ------------------------------------------------------------- public
    def set_category(self, category):
        self._category = category or ""
        cur = self.value()
        self._updating = True
        self._fill(self.family_combo,
                   [family_display(f) for f in TAXONOMY.families(self._category or None)],
                   family_display(cur["family"]))
        self._fill(self.genus_combo,
                   TAXONOMY.genera(self._category or None, cur["family"] or None),
                   cur["genus"])
        self._refill_species(cur["family"], cur["genus"])
        self._updating = False
        self._sync_level()

    def value(self):
        sp_disp = self.species_combo.currentText().strip()
        species, common = sp_disp, self._common
        m = _SPECIES_RE.match(sp_disp)
        if m:
            common, species = m.group(1).strip(), f"{m.group(2)} {m.group(3)}"
        return {"family": family_from_display(self.family_combo.currentText()),
                "genus": self.genus_combo.currentText().strip(),
                "species": species, "common": common}

    def set_value(self, family="", genus="", species="", common=""):
        self._updating = True
        self._common = common or ""
        if not family and genus:
            family = TAXONOMY.family_of_genus(genus)
        self._fill(self.family_combo,
                   [family_display(f) for f in TAXONOMY.families(self._category or None)],
                   family_display(family))
        self._fill(self.genus_combo,
                   TAXONOMY.genera(self._category or None, family or None), genus)
        rows = TAXONOMY.species_rows(self._category or None,
                                     family or None, genus or None)
        labels = [self._species_display(r) for r in rows]
        disp = ""
        if species:
            row = TAXONOMY.row_for_species(species)
            disp = self._species_display(row) if row else (
                f"{common} ({species})" if common else species)
        self._fill(self.species_combo, labels, disp)
        self._updating = False
        self._sync_level()

    def id_level(self):
        v = self.value()
        if v["species"]:
            return "species"
        if v["genus"]:
            return "genus"
        if v["family"]:
            return "family"
        return ""

    def clear(self):
        self.set_value("", "", "", "")
        self.find_edit.clear()
        self.changed.emit()

    def _sync_level(self):
        lvl = self.id_level()
        pretty = {"species": "Identified to species",
                  "genus": "Identified to genus",
                  "family": "Identified to family",
                  "": "Not identified"}[lvl]
        self.level_lbl.setText(pretty)

    def _add_species(self):
        fam = family_from_display(self.family_combo.currentText())
        gen = self.genus_combo.currentText().strip()
        text, ok = QInputDialog.getText(
            self, "Add species",
            "Binomial (e.g. Negaprion brevirostris) — optionally 'Common (Genus species)':")
        if not ok or not text.strip():
            return
        raw = text.strip()
        common, species = "", raw
        m = _SPECIES_RE.match(raw)
        if m:
            common, species = m.group(1).strip(), f"{m.group(2)} {m.group(3)}"
        parts = species.split()
        if not gen and parts and parts[0][:1].isupper():
            gen = parts[0]
        if not fam and gen:
            fam = TAXONOMY.family_of_genus(gen)
        row = {"category": self._category or "Other", "family": fam,
               "genus": gen, "species": species, "common": common}
        TAXONOMY.rows.append(row)
        TAXONOMY.reindex()
        try:
            write_taxonomy_csv(TAXONOMY.rows)
        except OSError as e:
            QMessageBox.warning(self, "Could not save taxonomy", str(e))
        self.set_value(fam, gen, species, common)
        self.changed.emit()


# --------------------------------------------------------------------------
# Label manager — add/remove/import/restore for behaviors, habitats, species
# --------------------------------------------------------------------------

class _ListEditor(QWidget):
    """A titled list with Add / Remove / Import CSV / Restore-default buttons.
    Edits an in-memory list; the owning dialog commits to CSV on OK."""

    def __init__(self, title, items, kind, importer, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.importer = importer     # callable(path, replace) or None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lab = QLabel(title)
        lab.setObjectName("BarTitle")
        lay.addWidget(lab)
        self.list = QListWidget()
        self.list.addItems(items)
        lay.addWidget(self.list, stretch=1)
        row = QHBoxLayout()
        for text, fn in (("Add", self._add), ("Remove", self._remove),
                         ("Import CSV", self._import), ("Restore default", self._restore)):
            b = QPushButton(text)
            b.setObjectName("Secondary")
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)

    def values(self):
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _set(self, items):
        self.list.clear()
        self.list.addItems(items)

    def _add(self):
        text, ok = QInputDialog.getText(self, "Add", f"New {self.kind}:")
        if ok and text.strip() and text.strip() not in self.values():
            self.list.addItem(text.strip())

    def _remove(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def _import(self):
        if self.importer is None:
            return
        p, _ = QFileDialog.getOpenFileName(self, "Import CSV", "", "CSV (*.csv)")
        if not p:
            return
        try:
            self.importer(p, replace=True)
            load_all_label_config()
            self._set(self._current_from_globals())
            QMessageBox.information(self, "Imported", f"Imported labels from:\n{p}")
        except Exception as e:
            QMessageBox.critical(self, "Import failed", str(e))

    def _restore(self):
        try:
            restore_default(self._restore_kind())
            load_all_label_config()
            self._set(self._current_from_globals())
        except Exception as e:
            QMessageBox.critical(self, "Restore failed", str(e))

    # hooks overridden per-instance
    def _current_from_globals(self):
        return self.values()

    def _restore_kind(self):
        return "behaviors"


class LabelManagerDialog(QDialog):
    """Manage Movement / Social / Habitat / Species label sets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage labels")
        self.resize(560, 460)
        v = QVBoxLayout(self)
        info = QLabel("Add or remove labels, import a CSV, or restore a category "
                      "to the built-in defaults. Changes apply when you click Save.")
        info.setObjectName("Subtle")
        info.setWordWrap(True)
        v.addWidget(info)

        tabs = QTabWidget()
        self.mov = _ListEditor("Movement behaviors", MOVEMENT_OPTIONS, "movement behavior",
                               import_behaviors_csv)
        self.mov._current_from_globals = lambda: list(MOVEMENT_OPTIONS)
        self.mov._restore_kind = lambda: "behaviors"
        self.soc = _ListEditor("Social behaviors", SOCIAL_OPTIONS, "social behavior",
                               import_behaviors_csv)
        self.soc._current_from_globals = lambda: list(SOCIAL_OPTIONS)
        self.soc._restore_kind = lambda: "behaviors"
        self.hab = _ListEditor("Habitats", HABITAT_OPTIONS, "habitat", import_habitats_csv)
        self.hab._current_from_globals = lambda: list(HABITAT_OPTIONS)
        self.hab._restore_kind = lambda: "habitats"
        tabs.addTab(self.mov, "Movement")
        tabs.addTab(self.soc, "Social")
        tabs.addTab(self.hab, "Habitats")
        tabs.addTab(self._species_tab(), "Species")
        v.addWidget(tabs, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("Secondary")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.clicked.connect(self._commit)
        btn_row.addWidget(cancel)
        btn_row.addWidget(save)
        v.addLayout(btn_row)

    def _species_tab(self):
        """Taxonomy tab — browse/edit by category, showing rank for each row."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self.sp_cat = QComboBox()
        self.sp_cat.addItems(SPECIES_CATEGORY_ORDER)
        self.sp_cat.currentTextChanged.connect(self._reload_species_list)
        top = QHBoxLayout()
        top.addWidget(QLabel("Category:"))
        top.addWidget(self.sp_cat, stretch=1)
        lay.addLayout(top)
        hint = QLabel("Import a CSV with columns: category, family, genus, "
                      "species, common  (extra columns ignored; a single column "
                      "of names also works).")
        hint.setObjectName("Subtle")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.sp_list = QListWidget()
        lay.addWidget(self.sp_list, stretch=1)
        row = QHBoxLayout()
        for text, fn in (("Add", self._sp_add), ("Remove", self._sp_remove),
                         ("Import CSV", self._sp_import),
                         ("Restore default", self._sp_restore)):
            b = QPushButton(text)
            b.setObjectName("Secondary")
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        # working copy of the taxonomy rows, committed on Save
        self._tax_work = [dict(r) for r in TAXONOMY.rows]
        self._reload_species_list()
        return w

    @staticmethod
    def _tax_row_text(r):
        if r.get("species"):
            base = f"{r['common']} ({r['species']})" if r.get("common") else r["species"]
            return f"{base}   · species"
        if r.get("genus"):
            return f"{r['genus']}   · genus"
        return f"{r.get('family','')}   · family"

    def _reload_species_list(self, *_):
        cat = self.sp_cat.currentText()
        self.sp_list.clear()
        self._tax_visible = [r for r in self._tax_work
                             if (r.get("category") or "Other") == cat]
        rank = {"family": 0, "genus": 1, "species": 2}
        self._tax_visible.sort(key=lambda r: (
            r.get("family", ""),
            rank["species" if r.get("species") else ("genus" if r.get("genus") else "family")],
            r.get("genus", ""), r.get("species", "")))
        for r in self._tax_visible:
            self.sp_list.addItem(self._tax_row_text(r))

    def _sp_add(self):
        cat = self.sp_cat.currentText()
        text, ok = QInputDialog.getText(
            self, "Add taxon",
            f"Add to {cat} — a family (Carcharhinidae), a genus (Negaprion sp.)\n"
            "or a species (Negaprion brevirostris / Lemon shark (Negaprion brevirostris)):")
        if not (ok and text.strip()):
            return
        raw = text.strip()
        r = {"category": cat, "family": "", "genus": "", "species": "", "common": ""}
        m = _SPECIES_RE.match(raw)
        if m:
            r["common"], r["genus"] = m.group(1).strip(), m.group(2)
            r["species"] = f"{m.group(2)} {m.group(3)}"
        elif raw.endswith(" sp."):
            r["genus"] = raw[:-4].strip()
        elif raw.endswith("idae"):
            r["family"] = raw
        else:
            parts = raw.split()
            if len(parts) >= 2 and parts[0][:1].isupper():
                r["genus"], r["species"] = parts[0], f"{parts[0]} {parts[1]}"
            else:
                r["species"] = raw
        if not r["family"] and r["genus"]:
            r["family"] = GENUS_FAMILY.get(r["genus"], "")
        self._tax_work.append(r)
        self._reload_species_list()

    def _sp_remove(self):
        for it in self.sp_list.selectedItems():
            i = self.sp_list.row(it)
            if 0 <= i < len(self._tax_visible):
                target = self._tax_visible[i]
                try:
                    self._tax_work.remove(target)
                except ValueError:
                    pass
        self._reload_species_list()

    def _sp_import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import taxonomy CSV", "", "CSV (*.csv)")
        if not p:
            return
        try:
            import_taxonomy_csv(p, replace=True)
            load_all_label_config()
            self._tax_work = [dict(r) for r in TAXONOMY.rows]
            self.sp_cat.blockSignals(True)
            self.sp_cat.clear()
            self.sp_cat.addItems(SPECIES_CATEGORY_ORDER)
            self.sp_cat.blockSignals(False)
            self._reload_species_list()
            QMessageBox.information(self, "Imported", f"Imported taxonomy from:\n{p}")
        except Exception as e:
            QMessageBox.critical(self, "Import failed", str(e))

    def _sp_restore(self):
        restore_default("taxonomy")
        load_all_label_config()
        self._tax_work = [dict(r) for r in TAXONOMY.rows]
        self._reload_species_list()

    def _commit(self):
        write_behaviors_csv(self.mov.values(), self.soc.values())
        write_habitats_csv(self.hab.values())
        write_taxonomy_csv(self._tax_work)
        load_all_label_config()
        self.accept()


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, video_path: str, fps: float, total_frames: int,
                 video_w: int, video_h: int,
                 is_frame_dir: bool = False,
                 temp_dir: Optional[str] = None,
                 store_video_path: Optional[str] = None,
                 preloaded_store: Optional[FeatureStore] = None):
        super().__init__()
        self.setWindowTitle("CTAG Annotator")

        self.video_path = video_path
        self.is_frame_dir = is_frame_dir
        self.temp_dir = temp_dir

        self.fps = float(fps) or 30.0
        self.total_frames = int(total_frames)
        self.video_w = int(video_w)
        self.video_h = int(video_h)

        # For frame directories, load frame list
        if is_frame_dir:
            self.frame_files = sorted([
                os.path.join(video_path, f) for f in os.listdir(video_path)
                if f.lower().endswith(('.jpg', '.jpeg'))
            ], key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            self.cap = None
            # Trust the values passed in; re-derive total_frames from files if needed
            if self.total_frames == 0:
                self.total_frames = len(self.frame_files)
        else:
            self.frame_files = None
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video_path}")

        self.current_frame_idx = 0
        self.playing = False
        self._labeling = True  # master switch: False = navigate-only
        # per-axis labeling switches — lets you re-scrub a section overwriting
        # only one axis (e.g. fix Habitat from Sand → Gravel without touching
        # Movement/Social/Visibility)
        self._label_axes = {"movement": True, "social": True,
                            "habitat": True, "visibility": True}

        # background preload of the next playlist video (path -> prep dict)
        self._preload: Dict[str, dict] = {}
        self._preload_inflight: Optional[str] = None

        # state — either use preloaded store or create fresh one
        store_path = store_video_path or video_path
        if preloaded_store is not None:
            self.store = preloaded_store
            # override path to match current video
            self.store.video_path = store_path
        else:
            self.store = FeatureStore(store_path, self.fps,
                                      self.total_frames, self.video_w, self.video_h)

        # scene-label selection (persist-until-changed) across all axes
        # Start every axis blank rather than pre-armed with the first option:
        # scrubbing a video before deliberately choosing a label should record
        # nothing, not silently paint "Burst swimming" over the whole clip.
        self._current_movement = None
        self._current_social = None
        self._current_habitat = None
        self._current_visibility = None
        self._type_counters: Dict[str, int] = {}

        # selection + editing
        self._selected_feature_idx: Optional[int] = None
        # zoom viewport (view rect of the full-res frame currently shown)
        self._zoom = 1.0
        self._zoom_cx = self.video_w / 2.0
        self._zoom_cy = self.video_h / 2.0
        self._view = None  # (vx, vy, vw, vh, pw, ph, ox, oy) set on each display

        # undo stack of state snapshots (discrete actions only)
        self._undo_stack: List[dict] = []
        self._undo_limit = 40

        # highlight-clip in/out marks
        self._clip_in: Optional[int] = None
        self._clip_out: Optional[int] = None

        # playback speed
        self._play_speed = 1.0

        # playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(max(1, int(1000 / self.fps)))

        # autosave: a periodic safety net PLUS a short debounce that fires
        # after any change, so no more than a couple of seconds of work is
        # ever at risk.
        self._dirty = False
        self._dirty_delay_ms = 2000
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(lambda: self._autosave("timer"))
        self.autosave_timer.start(60_000)  # periodic backstop

        self._dirty_timer = QTimer(self)
        self._dirty_timer.setSingleShot(True)
        self._dirty_timer.timeout.connect(lambda: self._autosave("change"))

        self._build_ui()
        self.seek_to(0)
        # Offer to resume a prior autosave for this source (unless preloaded).
        # Deferred so the main window is visible first — a modal dialog during
        # __init__ would block startup with the window not yet on screen.
        if preloaded_store is None:
            QTimer.singleShot(400, self._maybe_resume_autosave)

    # ------------------------------------------------------------------ UI
    def _make_segment_bar(self, title: str, options: List[str], on_select,
                          rows: int = 1, on_add=None, allow_none=False):
        """Build a single-row segmented button bar.

        Returns (container, btns_dict, append_fn).  append_fn(label) adds a new
        option button live (used by the in-app "＋ Add" flow).  If on_add is
        given, a trailing "＋" button invokes it.
        """
        container = FlowBar()

        try:
            from PyQt6.QtWidgets import QButtonGroup
        except ImportError:
            from PyQt5.QtWidgets import QButtonGroup

        group = QButtonGroup(self)
        group.setExclusive(True)
        btns: Dict[str, QToolButton] = {}

        # FlowLayout (not QHBoxLayout) so the bar wraps instead of forcing the
        # whole window wider than the screen on small laptops.
        fl = FlowLayout(margin=0, spacing=5)
        lab = QLabel(title)
        lab.setObjectName("BarTitle")
        lab.setMinimumWidth(66)
        fl.addWidget(lab)

        # "None" clears this axis: selecting it and re-playing/scrubbing over a
        # stretch erases those labels, which is the fix for having scrubbed
        # through a video with labeling accidentally left on.
        if allow_none:
            nb = QToolButton()
            nb.setText("None")
            nb.setCheckable(True)
            nb.setProperty("segmented", True)
            nb.setToolTip("Leave this axis unlabeled.\nSelect None, then play or "
                          "scrub back over a section to erase it.")
            nb.clicked.connect(lambda checked: on_select(None))
            group.addButton(nb)
            btns[None] = nb
            fl.addWidget(nb)

        def _make_btn(opt):
            b = QToolButton()
            b.setText(opt)
            b.setCheckable(True)
            b.setProperty("segmented", True)
            b.clicked.connect(lambda checked, t=opt: on_select(t))
            group.addButton(b)
            btns[opt] = b
            return b

        def _add_option_button(opt):
            b = _make_btn(opt)
            # keep new options ahead of the trailing "＋" button
            fl.insertWidget(fl.count() - (1 if on_add is not None else 0), b)
            return b

        for opt in options:
            fl.addWidget(_make_btn(opt))

        if on_add is not None:
            add_btn = QToolButton()
            add_btn.setText("＋")
            add_btn.setProperty("segmented", True)
            add_btn.setToolTip("Add a new option")
            add_btn.clicked.connect(lambda: on_add())
            fl.addWidget(add_btn)

        container.setLayout(fl)
        # A heightForWidth layout is only honoured if the *size policy* also
        # declares it — without this the parent layout ignores the wrapped
        # height and the bars overlap each other.
        try:
            sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        except AttributeError:
            sp = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        sp.setHeightForWidth(True)
        container.setSizePolicy(sp)
        return container, btns, _add_option_button

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # -------------------------------- video display
        self.video_label = VideoLabel()
        align_flag = Qt.AlignmentFlag.AlignCenter if _QT6 else Qt.AlignCenter
        self.video_label.setAlignment(align_flag)
        self.video_label.setMinimumSize(QSize(360, 240))
        try:
            _exp = QSizePolicy.Policy.Expanding
        except AttributeError:
            _exp = QSizePolicy.Expanding
        self.video_label.setSizePolicy(_exp, _exp)

        # Scene-label bars (below video): two behavior axes + habitat + visibility
        self.movement_bar, self._movement_btns, self._movement_append = \
            self._make_segment_bar(
                "Movement:", MOVEMENT_OPTIONS, self._on_movement_selected,
                on_add=lambda: self._add_behavior_dialog("movement"),
                allow_none=True)
        self.social_bar, self._social_btns, self._social_append = \
            self._make_segment_bar(
                "Social:", SOCIAL_OPTIONS, self._on_social_selected,
                on_add=lambda: self._add_behavior_dialog("social"),
                allow_none=True)
        self.habitat_bar, self._habitat_btns, self._habitat_append = \
            self._make_segment_bar(
                "Habitat:", HABITAT_OPTIONS, self._on_habitat_selected,
                on_add=self._add_habitat_dialog, allow_none=True)
        self.visibility_bar, self._visibility_btns, _ = self._make_segment_bar(
            "Visibility:", VISIBILITY_OPTIONS, self._on_visibility_selected,
            allow_none=True
        )
        # Conspecific shark logger lives ON the social bar (sharks are a social
        # observation, not a fish-ID annotation)
        self._shark_logger = self._make_shark_logger()
        self.social_bar.layout().addWidget(self._shark_logger)
        # initial highlight
        self._movement_btns[self._current_movement].setChecked(True)
        self._social_btns[self._current_social].setChecked(True)
        self._habitat_btns[self._current_habitat].setChecked(True)
        self._visibility_btns[self._current_visibility].setChecked(True)

        # The four label bars live in their own height-capped scroll area. They
        # wrap on narrow windows, and capping the height means however much they
        # wrap they can never force the whole window taller than the screen.
        bars_host = QWidget()
        bars_col = QVBoxLayout(bars_host)
        bars_col.setSpacing(3)
        bars_col.setContentsMargins(0, 0, 0, 0)
        for bar in (self.movement_bar, self.social_bar,
                    self.habitat_bar, self.visibility_bar):
            bars_col.addWidget(bar)
        bars_col.addStretch(0)   # soak up slack instead of stretching the bars
        self._bars_col = bars_col

        # Each FlowBar reports the height its rows actually need at the width
        # it is given, so the strip sizes itself exactly — no scroll area, no
        # scrollbar on a row of buttons, and no reserved dead space stealing
        # room from the video.
        try:
            sp = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        except AttributeError:
            sp = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        bars_host.setSizePolicy(sp)
        self._bars_host = bars_host

        video_col = QVBoxLayout()
        video_col.setSpacing(6)
        video_col.setContentsMargins(0, 0, 0, 0)
        video_col.addWidget(self.video_label, stretch=1)
        video_col.addWidget(bars_host)
        self._video_col = video_col

        video_wrap = QWidget()
        video_wrap.setLayout(video_col)

        # -------------------------------- right panel (feature list — read-only)
        # feature list (populated from JSON if available; cannot add without tracking)
        def _tidy_list(lw):
            """Elide long rows instead of growing a horizontal scrollbar."""
            try:
                lw.setTextElideMode(Qt.TextElideMode.ElideRight)
                lw.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            except AttributeError:
                lw.setTextElideMode(Qt.ElideRight)
                lw.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            lw.setWordWrap(False)
            return lw

        self.feat_list = _tidy_list(QListWidget())
        self.feat_list.currentItemChanged.connect(self._on_feature_selected)
        self._tidy_list = _tidy_list

        fa = QHBoxLayout()
        del_btn = QPushButton("Delete")
        del_btn.setObjectName("Secondary")
        del_btn.clicked.connect(self._delete_feature)
        # Edits to a selected box apply instantly (see _live_edit), so there is
        # no "Update" step to forget — just a way to leave edit mode.
        self.done_edit_btn = QPushButton("Done editing")
        self.done_edit_btn.setObjectName("Secondary")
        self.done_edit_btn.clicked.connect(self._end_edit)
        fa.addWidget(del_btn)
        fa.addWidget(self.done_edit_btn)

        # Says which mode the fields below are in: settings for the NEXT box,
        # or live edits to an existing one.
        self.edit_banner = QLabel()
        self.edit_banner.setWordWrap(True)

        self.name_edit = QLineEdit("feature")

        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_row.addWidget(QLabel("Name:"))
        name_row.addWidget(self.name_edit, stretch=1)

        # export
        exp_btn = QPushButton("Export Video")
        exp_btn.clicked.connect(self._export_video)
        plots_btn = QPushButton("Export Plots")
        plots_btn.setObjectName("Secondary")
        plots_btn.clicked.connect(self._export_plots_only)
        excel_btn = QPushButton("Export Excel")
        excel_btn.setObjectName("Secondary")
        excel_btn.clicked.connect(self._export_excel)
        json_btn = QPushButton("Save JSON")
        json_btn.setObjectName("Secondary")
        json_btn.clicked.connect(self._save_json)
        load_json_btn = QPushButton("Load JSON")
        load_json_btn.setObjectName("Secondary")
        load_json_btn.clicked.connect(self._load_json)

        # Combos must be allowed to shrink below their content width, otherwise
        # the whole right panel gets forced wider than its container and clips.
        def _shrinkable(combo, chars=10):
            try:
                combo.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            except AttributeError:
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(chars)
            return combo

        # Category combo (derived from the loaded species config)
        first_cat = SPECIES_CATEGORY_ORDER[0] if SPECIES_CATEGORY_ORDER else "Shark"
        self.category_combo = _shrinkable(QComboBox())
        self.category_combo.addItems(SPECIES_CATEGORY_ORDER)

        category_row = QHBoxLayout()
        category_row.addWidget(QLabel("Category:"))
        category_row.addWidget(self.category_combo, stretch=1)

        # Taxonomy picker: find-box + Family/Genus/Species, so an annotator can
        # stop at whatever rank they're confident about.
        self.taxon = TaxonomyPicker()
        self.taxon.set_category(first_cat)
        self.category_combo.currentTextChanged.connect(self._on_category_changed)

        # Count spinbox + confidence.  (No sex fields here — conspecific sharks
        # are logged on the Social bar, not as fish-ID annotations.)
        self.count_spin = QSpinBox()
        self.count_spin.setMinimum(1)
        self.count_spin.setMaximum(9999)
        self.count_spin.setValue(1)
        self.count_spin.setMinimumWidth(52)
        self.confidence_combo = _shrinkable(QComboBox(), chars=6)
        self.confidence_combo.addItems(CONFIDENCE_OPTIONS)

        count_row = QHBoxLayout()
        count_row.setSpacing(6)
        count_row.addWidget(QLabel("Count:"))
        count_row.addWidget(self.count_spin, stretch=1)
        count_row.addWidget(QLabel("Conf:"))
        count_row.addWidget(self.confidence_combo, stretch=1)

        # Draw bbox toggle button
        self.bbox_btn = QPushButton("Draw Bbox")
        self.bbox_btn.setCheckable(True)
        self.bbox_btn.toggled.connect(self._bbox_mode_toggled)

        # Wire bbox signals (two-click draw, corner edit, ctrl-wheel zoom)
        self.video_label.bboxDrawn.connect(self._on_bbox)
        self.video_label.bboxEdited.connect(self._on_bbox_edited)
        self.video_label.zoomRequested.connect(self._on_wheel_zoom)
        self.video_label.pointClicked.connect(self._on_video_click)
        # Live editing of the selected box
        self._loading_edit = False     # True while WE fill the fields
        self._edit_undo_for = None     # feature idx whose edit burst has an undo point
        self._new_box_fields = None    # next-box settings stashed during an edit
        self.name_edit.textEdited.connect(lambda _t: self._live_edit())
        self.category_combo.currentTextChanged.connect(lambda _t: self._live_edit())
        self.taxon.changed.connect(self._live_edit)
        self.count_spin.valueChanged.connect(lambda _v: self._live_edit())
        self.confidence_combo.currentTextChanged.connect(lambda _t: self._live_edit())
        self._update_edit_banner()
        self.video_label.firstCornerPlaced.connect(
            lambda: self.statusBar().showMessage(
                "Now click the opposite corner (right-click to cancel)."))

        # ---- Highlight clips ----
        self.clip_range_lbl = QLabel("In —  ·  Out —")
        self.clip_range_lbl.setObjectName("Subtle")
        mark_in_btn = QPushButton("Mark In [")
        mark_in_btn.setObjectName("Secondary")
        mark_in_btn.clicked.connect(self._mark_in)
        mark_out_btn = QPushButton("Mark Out ]")
        mark_out_btn.setObjectName("Secondary")
        mark_out_btn.clicked.connect(self._mark_out)
        clip_mark_row = QHBoxLayout()
        clip_mark_row.addWidget(mark_in_btn)
        clip_mark_row.addWidget(mark_out_btn)

        self.clips_list = _tidy_list(QListWidget())
        self.clips_list.setMaximumHeight(100)
        self.clips_list.itemDoubleClicked.connect(self._on_clip_activated)
        save_clip_btn = QPushButton("Save Clip")
        save_clip_btn.setObjectName("Secondary")
        save_clip_btn.clicked.connect(self._save_clip)
        export_clip_btn = QPushButton("Export Clip")
        export_clip_btn.clicked.connect(self._export_clip)
        del_clip_btn = QPushButton("Delete Clip")
        del_clip_btn.setObjectName("Secondary")
        del_clip_btn.clicked.connect(self._delete_clip)
        clip_btn_row = QHBoxLayout()
        clip_btn_row.addWidget(save_clip_btn)
        clip_btn_row.addWidget(del_clip_btn)

        # ---- Conspecific shark log (fed by the Social bar's ♂/♀/?+Log) ----
        self.shark_log_list = _tidy_list(QListWidget())
        self.shark_log_list.setMaximumHeight(88)
        self.shark_log_list.itemDoubleClicked.connect(self._on_shark_entry_activated)
        del_shark_btn = QPushButton("Delete Entry")
        del_shark_btn.setObjectName("Secondary")
        del_shark_btn.clicked.connect(self._delete_shark_entry)

        # ---- Timestamped comments ----
        self.notes_list = _tidy_list(QListWidget())
        self.notes_list.setMaximumHeight(140)
        self.notes_list.itemDoubleClicked.connect(self._on_note_activated)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("Comment at current frame… e.g. 'ID this fish later'")
        self.note_edit.returnPressed.connect(self._add_note)
        add_note_btn = QPushButton("Add Comment")
        add_note_btn.setObjectName("Secondary")
        add_note_btn.clicked.connect(self._add_note)
        del_note_btn = QPushButton("Delete Comment")
        del_note_btn.setObjectName("Secondary")
        del_note_btn.clicked.connect(self._delete_note)
        note_input_row = QHBoxLayout()
        note_input_row.addWidget(self.note_edit, stretch=1)
        note_input_row.addWidget(add_note_btn)
        note_btn_row = QHBoxLayout()
        note_btn_row.addWidget(del_note_btn)

        # whole-video note (applies to the entire video, not a frame)
        self.video_note_edit = QLineEdit()
        self.video_note_edit.setPlaceholderText("Whole-video note (applies to the entire video)…")
        self.video_note_edit.editingFinished.connect(self._on_video_note_changed)

        # Pack into a "card"
        card_layout = QVBoxLayout()
        card_layout.addWidget(self.edit_banner)
        card_layout.addLayout(category_row)
        card_layout.addWidget(self.taxon)
        card_layout.addLayout(count_row)
        card_layout.addLayout(name_row)
        card_layout.addWidget(self.bbox_btn)
        card_layout.addWidget(QLabel("Boxes (click one to jump to it and edit):"))
        card_layout.addWidget(self.feat_list, stretch=1)
        card_layout.addLayout(fa)
        card_layout.addWidget(QLabel("Shark log (from Social bar, double-click to jump):"))
        card_layout.addWidget(self.shark_log_list)
        card_layout.addWidget(del_shark_btn)
        card_layout.addWidget(QLabel("Comments (double-click to jump):"))
        card_layout.addWidget(self.notes_list)
        card_layout.addLayout(note_input_row)
        card_layout.addLayout(note_btn_row)
        card_layout.addWidget(QLabel("Whole-video note:"))
        card_layout.addWidget(self.video_note_edit)
        card_layout.addWidget(QLabel("Highlight clips (I / O to mark):"))
        card_layout.addWidget(self.clip_range_lbl)
        card_layout.addLayout(clip_mark_row)
        card_layout.addWidget(self.clips_list)
        card_layout.addLayout(clip_btn_row)
        card_layout.addWidget(export_clip_btn)
        card_layout.addWidget(load_json_btn)
        card_layout.addWidget(exp_btn)
        card_layout.addWidget(plots_btn)
        card_layout.addWidget(excel_btn)
        card_layout.addWidget(json_btn)

        card = QFrame()
        card.setObjectName("Card")
        card_v = QVBoxLayout()
        card_v.setContentsMargins(12, 12, 12, 12)
        card_v.setSpacing(8)
        title = QLabel("Fish labeler")
        title.setObjectName("Title")
        subtitle = QLabel("Label movement, social, habitat & visibility. "
                          "Draw bboxes to annotate animals; add notes at any frame.")
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        card_v.addWidget(title)
        card_v.addWidget(subtitle)
        card_v.addLayout(card_layout)
        card.setLayout(card_v)

        rw = QScrollArea()
        rw.setWidget(card)
        rw.setWidgetResizable(True)
        rw.setFrameShape(QFrame.Shape.NoFrame if _QT6 else QFrame.NoFrame)
        hbar_off = (Qt.ScrollBarPolicy.ScrollBarAlwaysOff if _QT6
                    else Qt.ScrollBarAlwaysOff)
        rw.setHorizontalScrollBarPolicy(hbar_off)
        rw.setMaximumWidth(400)
        rw.setMinimumWidth(290)
        self._right_panel = rw

        # ---- playback controls ----
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setObjectName("Secondary")

        self.back_btn = QPushButton("◀")
        self.back_btn.setObjectName("Secondary")
        self.back_btn.clicked.connect(lambda: self.seek_to(self.current_frame_idx - 1))

        self.fwd_btn = QPushButton("▶")
        self.fwd_btn.setObjectName("Secondary")
        self.fwd_btn.clicked.connect(lambda: self.seek_to(self.current_frame_idx + 1))

        self.frame_lbl = QLabel()
        self.frame_lbl.setObjectName("Subtle")
        self.frame_lbl.setMinimumWidth(150)

        # Speed combo
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)

        self.label_mode_btn = QPushButton("Labeling: ON")
        self.label_mode_btn.setCheckable(True)
        self.label_mode_btn.setChecked(True)
        self.label_mode_btn.setObjectName("Secondary")
        self.label_mode_btn.setStyleSheet(
            "QPushButton:checked { background:#16A34A; color:#FFFFFF; border:none; }"
            "QPushButton:!checked { background:#6B7280; color:#FFFFFF; border:none; }"
        )
        self.label_mode_btn.clicked.connect(self._toggle_label_mode)

        # Per-axis labeling checkboxes: untick an axis to protect it while
        # re-scrubbing (e.g. fix only Habitat over an already-labeled section)
        self._axis_checkboxes = {}
        axis_row = QHBoxLayout()
        axis_row.setSpacing(4)
        for key, short in (("movement", "Mov"), ("social", "Soc"),
                           ("habitat", "Hab"), ("visibility", "Vis")):
            cb = QCheckBox(short)
            cb.setChecked(True)
            cb.setToolTip(f"Write {key} labels while playing/scrubbing")
            cb.toggled.connect(lambda on, k=key: self._on_axis_toggle(k, on))
            self._axis_checkboxes[key] = cb
            axis_row.addWidget(cb)

        # Zoom controls
        zoom_out_btn = QPushButton("－")
        zoom_out_btn.setObjectName("Secondary")
        zoom_out_btn.setToolTip("Zoom out (Ctrl/⌘ + wheel)")
        zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_in_btn = QPushButton("＋")
        zoom_in_btn.setObjectName("Secondary")
        zoom_in_btn.setToolTip("Zoom in (Ctrl/⌘ + wheel)")
        zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_reset_btn = QPushButton("Fit")
        zoom_reset_btn.setObjectName("Secondary")
        zoom_reset_btn.clicked.connect(self._zoom_reset)
        self.zoom_lbl = QLabel("1.0×")
        self.zoom_lbl.setObjectName("Subtle")

        self.focus_btn = QPushButton("Focus")
        self.focus_btn.setCheckable(True)
        self.focus_btn.setObjectName("Secondary")
        self.focus_btn.setToolTip("Hide the panels and just watch the video (F)")
        self.focus_btn.clicked.connect(lambda: self._toggle_focus_mode())

        # Playlist switcher (multiple videos / folder loading)
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(110)
        self.video_combo.currentIndexChanged.connect(self._on_playlist_changed)

        # Bottom control strip also uses FlowLayout: on a narrow window it wraps
        # to a second row instead of forcing the window wider than the display.
        axis_host = QWidget()
        axis_host.setLayout(axis_row)

        speed_host = QWidget()
        speed_hl = QHBoxLayout(speed_host)
        speed_hl.setContentsMargins(0, 0, 0, 0)
        speed_hl.setSpacing(4)
        speed_hl.addWidget(QLabel("Speed:"))
        speed_hl.addWidget(self.speed_combo)

        ctrls = FlowLayout(margin=0, spacing=7)
        for w in (self.back_btn, self.play_btn, self.fwd_btn, self.frame_lbl,
                  self.video_combo, zoom_out_btn, self.zoom_lbl, zoom_in_btn,
                  zoom_reset_btn, self.focus_btn, self.label_mode_btn,
                  axis_host, speed_host):
            ctrls.addWidget(w)

        # ---- bottom timeline scrubber ----
        self.timeline = AnnotationTimeline(self.store, self.fps, self.total_frames)
        self.timeline.frameSelected.connect(self._on_slider_seek)

        # ---- menu bar ----
        self._build_menu()

        # ---- keyboard shortcuts ----
        # Window-wide (modifier or arrow keys — safe around text fields)
        def _sc(seq, fn):
            QShortcut(QKeySequence(seq), self).activated.connect(fn)

        _sc(Qt.Key.Key_Left if _QT6 else Qt.Key_Left,
            lambda: self.seek_to(self.current_frame_idx - 1))
        _sc(Qt.Key.Key_Right if _QT6 else Qt.Key_Right,
            lambda: self.seek_to(self.current_frame_idx + 1))
        _sc("Ctrl+Z", self._undo)      # Ctrl+Z
        _sc("Meta+Z", self._undo)      # ⌘Z on macOS
        _sc("Ctrl++", self._zoom_in)
        _sc("Ctrl+=", self._zoom_in)
        _sc("Ctrl+-", self._zoom_out)
        _sc("Ctrl+I", self._mark_in)
        _sc("Ctrl+O", self._mark_out)
        _sc("F1", self._show_help)
        _sc("Ctrl+S", self._save_json)
        _sc("Ctrl+E", self._export_clip)
        _sc("Ctrl+Right", lambda: self.seek_to(self.current_frame_idx + 10))
        _sc("Ctrl+Left", lambda: self.seek_to(self.current_frame_idx - 10))
        # Bare-key shortcuts (Space, B, C) and Delete are handled in
        # keyPressEvent so they never hijack typing in a text field.

        # ---- assemble ----
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(video_wrap, stretch=1)
        top.addWidget(rw, stretch=0)

        lay = QVBoxLayout()
        lay.setContentsMargins(6, 6, 6, 4)
        lay.setSpacing(5)
        lay.addLayout(top, stretch=1)
        lay.addWidget(self.timeline)
        lay.addLayout(ctrls)
        central.setLayout(lay)

        # Focus mode + remembered preferences
        self._focus_mode = False
        self._focus_hidden = []
        self._apply_saved_settings()

        # Qt derives a minimum from the worst-case wrapping of the label bars
        # (very narrow window -> many rows) and would otherwise refuse to shrink
        # below ~1100px tall, which is more than a 13" display has. The bars
        # reflow fine at any size, so state the real floor explicitly.
        self.setMinimumSize(QSize(660, 540))

        self.statusBar().showMessage("Ready.")
        self._refresh_features()
        self._refresh_notes()
        self._refresh_clips()
        self._refresh_shark_log()
        self._sync_video_note()
        self._update_clip_label()
        self._update_zoom_label()

    def _build_menu(self):
        try:
            from PyQt6.QtGui import QAction
        except ImportError:
            from PyQt5.QtWidgets import QAction
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        a1 = QAction("Open Video…", self)
        a1.triggered.connect(self._open_video_dialog)
        a2 = QAction("Open Frames Folder…", self)
        a2.triggered.connect(self._open_frames_dialog)
        a3 = QAction("Open Folder of Videos…", self)
        a3.triggered.connect(self._open_folder_dialog)
        for a in (a1, a2, a3):
            file_menu.addAction(a)
        file_menu.addSeparator()
        a_save = QAction("Save JSON…", self)
        a_save.triggered.connect(self._save_json)
        file_menu.addAction(a_save)
        a_reveal = QAction("Open Local Annotations Folder", self)
        a_reveal.triggered.connect(self._open_local_annotations_folder)
        file_menu.addAction(a_reveal)

        labels_menu = mb.addMenu("&Labels")
        a_manage = QAction("Manage labels (add / remove / import CSV)…", self)
        a_manage.triggered.connect(self._manage_labels)
        labels_menu.addAction(a_manage)
        a_reveal_cfg = QAction("Open Config Folder (CSV files)", self)
        a_reveal_cfg.triggered.connect(self._open_config_folder)
        labels_menu.addAction(a_reveal_cfg)

        help_menu = mb.addMenu("&Help")
        a_help = QAction("Keyboard & usage help", self)
        a_help.triggered.connect(self._show_help)
        help_menu.addAction(a_help)

    # ------------------------------------------------------------ env/sub
    def _bbox_mode_toggled(self, on: bool):
        if on and self._selected_feature_idx is not None:
            self._end_edit()           # fields go back to next-box settings
        self.video_label.set_bbox_mode(on)
        self._update_edit_rect(self.current_frame_idx)
        if on:
            self.bbox_btn.setText("Click 2 corners…")
            self.bbox_btn.setStyleSheet(
                "QPushButton { background:#4F46E5; color:#FFFFFF; border:none; }")
            self.statusBar().showMessage(
                "Click two corners on the video to draw a box (right-click cancels).")
        else:
            self.bbox_btn.setText("Draw Bbox")
            self.bbox_btn.setStyleSheet("")
            self.statusBar().showMessage("")

    def _on_bbox(self, p1, p2):
        m1 = self._map(p1.x(), p1.y())
        m2 = self._map(p2.x(), p2.y())
        if m1 is None or m2 is None:
            return
        x1, y1 = min(m1[0], m2[0]), min(m1[1], m2[1])
        x2, y2 = max(m1[0], m2[0]), max(m1[1], m2[1])
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        species_category = self.category_combo.currentText() or "Other"
        tax = self.taxon.value()
        confidence = self.confidence_combo.currentText().strip()
        count = self.count_spin.value()
        name = (self.name_edit.text() or "").strip()
        if not name:
            n = self._type_counters.get(species_category, 1)
            name = f"{species_category.lower().replace(' ', '_')}_{n}"
            self._type_counters[species_category] = n + 1
            self.name_edit.setText(name)

        fidx = self.current_frame_idx
        self._push_undo()
        feat = TrackedFeature(
            name=name,
            init_frame=fidx,
            end_frame=fidx,
            init_type="bbox",
            init_coords=[x1, y1, x2, y2],
            count=count,
            species_category=species_category,
            species=tax["species"],
            family=tax["family"],
            genus=tax["genus"],
            common=tax["common"],
            confidence=confidence,
        )
        self.store.add_feature(feat, {}, {fidx: (x1, y1, x2, y2)})
        self._refresh_features()
        self._display_frame(fidx)
        if hasattr(self, "timeline"):
            self.timeline.update()
        # Report the ID at whatever rank was actually recorded (species,
        # genus or family). NOTE: read this off the feature, not a local —
        # a stale local here previously raised NameError inside the Qt slot
        # and took the whole app down mid-annotation.
        id_txt = feat.id_label()
        if id_txt and id_txt != species_category:
            self.statusBar().showMessage(
                f"Added {species_category} ({id_txt}) ×{count} '{name}' on frame {fidx}."
            )
        else:
            self.statusBar().showMessage(
                f"Added {species_category} ×{count} '{name}' on frame {fidx}."
            )
        # clear name + count so the next animal starts fresh
        self.name_edit.clear()
        self.count_spin.setValue(1)
        # auto-disarm so the mode state is always explicit (arm → draw → off)
        self.bbox_btn.setChecked(False)

    def _map(self, lx: int, ly: int) -> Optional[Tuple[int, int]]:
        """Label-widget coords → full-res video-frame coords (zoom-aware)."""
        if not self._view:
            return None
        vx, vy, vw, vh, pw, ph, ox, oy = self._view
        x, y = lx - ox, ly - oy
        if x < 0 or y < 0 or x >= pw or y >= ph:
            return None
        fx = int(vx + x * vw / pw)
        fy = int(vy + y * vh / ph)
        fx = max(0, min(fx, self.video_w - 1))
        fy = max(0, min(fy, self.video_h - 1))
        return (fx, fy)

    def _video_to_label(self, vx: float, vy: float) -> Tuple[float, float]:
        """Full-res video coords → label-widget coords (zoom-aware)."""
        if not self._view:
            return (vx, vy)
        rvx, rvy, rvw, rvh, pw, ph, ox, oy = self._view
        lx = ox + (vx - rvx) * pw / max(1e-6, rvw)
        ly = oy + (vy - rvy) * ph / max(1e-6, rvh)
        return (lx, ly)

    def _on_movement_selected(self, label):
        """label=None means 'unlabeled' — selecting None and scrubbing back
        over a section erases it."""
        self._push_undo()
        self._current_movement = label
        self.store.set_movement(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Movement: {label or '— none (erasing) —'}")

    def _on_social_selected(self, label):
        """label=None means 'unlabeled' — selecting None and scrubbing back
        over a section erases it."""
        self._push_undo()
        self._current_social = label
        self.store.set_social(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self._refresh_shark_log()   # spans derive from social segments
        self.statusBar().showMessage(f"Social: {label or '— none (erasing) —'}")

    def _on_habitat_selected(self, label):
        """label=None means 'unlabeled' — selecting None and scrubbing back
        over a section erases it."""
        self._push_undo()
        self._current_habitat = label
        self.store.set_habitat(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Habitat: {label or '— none (erasing) —'}")

    def _on_visibility_selected(self, label):
        """label=None means 'unlabeled' — selecting None and scrubbing back
        over a section erases it."""
        self._push_undo()
        self._current_visibility = label
        self.store.set_visibility(self.current_frame_idx, label)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Visibility: {label or '— none (erasing) —'}")

    def _add_behavior_dialog(self, axis: str):
        """Prompt for a new movement/social behavior, persist to CSV, add live."""
        try:
            from PyQt6.QtWidgets import QInputDialog
        except ImportError:
            from PyQt5.QtWidgets import QInputDialog
        title = "Add movement behavior" if axis == "movement" else "Add social behavior"
        text, ok = QInputDialog.getText(self, title, "New label:")
        if not ok:
            return
        label = (text or "").strip()
        if not label:
            return
        options = MOVEMENT_OPTIONS if axis == "movement" else SOCIAL_OPTIONS
        btns = self._movement_btns if axis == "movement" else self._social_btns
        if label in options:
            btns[label].setChecked(True)
            (self._on_movement_selected if axis == "movement"
             else self._on_social_selected)(label)
            return
        options.append(label)
        append_behavior_to_csv(axis, label)
        append_fn = self._movement_append if axis == "movement" else self._social_append
        append_fn(label)
        # keep the timeline color maps in sync with the new option
        if hasattr(self, "timeline"):
            palette = FEATURE_COLORS
            if axis == "movement":
                i = len(MOVEMENT_OPTIONS) - 1
                self.timeline._movement_colors[label] = QColor(*palette[i % len(palette)])
            else:
                i = len(SOCIAL_OPTIONS) - 1
                self.timeline._social_colors[label] = QColor(*palette[(i + 3) % len(palette)])
        btns[label].setChecked(True)
        (self._on_movement_selected if axis == "movement"
         else self._on_social_selected)(label)

    def _add_habitat_dialog(self):
        text, ok = QInputDialog.getText(self, "Add habitat", "New habitat:")
        if not ok:
            return
        label = (text or "").strip()
        if not label:
            return
        if label not in HABITAT_OPTIONS:
            HABITAT_OPTIONS.append(label)
            write_habitats_csv(HABITAT_OPTIONS)
            self._habitat_append(label)
            if hasattr(self, "timeline"):
                i = len(HABITAT_OPTIONS) - 1
                self.timeline._habitat_colors[label] = QColor(
                    *FEATURE_COLORS[i % len(FEATURE_COLORS)])
        if label in self._habitat_btns:
            self._habitat_btns[label].setChecked(True)
            self._on_habitat_selected(label)

    def _manage_labels(self):
        dlg = LabelManagerDialog(self)
        if dlg.exec():
            self._rebuild_label_bars()
            self.statusBar().showMessage("Labels updated.")

    def _rebuild_label_bars(self):
        """Recreate the movement/social/habitat bars + species combos after the
        label config changed (Manage labels / import / restore)."""
        # remember current selections
        cur = (self._current_movement, self._current_social, self._current_habitat)
        # detach the shark logger so deleting the old social bar can't kill it
        self._shark_logger.setParent(None)
        for bar in (self.movement_bar, self.social_bar, self.habitat_bar):
            self._bars_col.removeWidget(bar)
            bar.setParent(None)
            bar.deleteLater()
        self.movement_bar, self._movement_btns, self._movement_append = \
            self._make_segment_bar("Movement:", MOVEMENT_OPTIONS,
                                   self._on_movement_selected,
                                   on_add=lambda: self._add_behavior_dialog("movement"),
                                   allow_none=True)
        self.social_bar, self._social_btns, self._social_append = \
            self._make_segment_bar("Social:", SOCIAL_OPTIONS,
                                   self._on_social_selected,
                                   on_add=lambda: self._add_behavior_dialog("social"),
                                   allow_none=True)
        self.habitat_bar, self._habitat_btns, self._habitat_append = \
            self._make_segment_bar("Habitat:", HABITAT_OPTIONS,
                                   self._on_habitat_selected,
                                   on_add=self._add_habitat_dialog,
                                   allow_none=True)
        self.social_bar.layout().addWidget(self._shark_logger)  # re-attach
        self._bars_col.insertWidget(0, self.movement_bar)
        self._bars_col.insertWidget(1, self.social_bar)
        self._bars_col.insertWidget(2, self.habitat_bar)
        # keep valid current selections, else fall back to first option
        self._current_movement = cur[0] if (cur[0] is None or cur[0] in MOVEMENT_OPTIONS) else MOVEMENT_OPTIONS[0]
        self._current_social = cur[1] if (cur[1] is None or cur[1] in SOCIAL_OPTIONS) else SOCIAL_OPTIONS[0]
        self._current_habitat = cur[2] if (cur[2] is None or cur[2] in HABITAT_OPTIONS) else HABITAT_OPTIONS[0]
        for d, k in ((self._movement_btns, self._current_movement),
                     (self._social_btns, self._current_social),
                     (self._habitat_btns, self._current_habitat)):
            b = d.get(k)
            if b is not None:
                b.setChecked(True)
        # rebuild timeline colors + category/species combos
        self._rebuild_timeline_colors()
        self._rebuild_category_combo()
        if hasattr(self, "timeline"):
            self.timeline.update()

    def _rebuild_timeline_colors(self):
        if not hasattr(self, "timeline"):
            return
        tl = self.timeline
        pal = FEATURE_COLORS
        tl._movement_colors = {opt: QColor(*pal[i % len(pal)])
                               for i, opt in enumerate(MOVEMENT_OPTIONS)}
        tl._movement_colors[None] = QColor(203, 213, 225)
        tl._social_colors = {opt: QColor(*pal[(i + 3) % len(pal)])
                             for i, opt in enumerate(SOCIAL_OPTIONS)}
        for j, opt in enumerate(sorted(RETIRED_SOCIAL)):
            tl._social_colors.setdefault(
                opt, QColor(*pal[(len(SOCIAL_OPTIONS) + j + 3) % len(pal)]))
        tl._social_colors[None] = QColor(203, 213, 225)
        known = {"Mangrove": QColor(16, 185, 129), "Rocky reef": QColor(14, 165, 233),
                 "Sandy bottom": QColor(245, 158, 11), "Gravel": QColor(156, 163, 175),
                 "Mud": QColor(120, 100, 80)}
        tl._habitat_colors = {None: QColor(203, 213, 225)}
        for i, h in enumerate(HABITAT_OPTIONS):
            tl._habitat_colors[h] = known.get(h, QColor(*pal[i % len(pal)]))

    def _rebuild_category_combo(self):
        if not hasattr(self, "category_combo"):
            return
        cur = self.category_combo.currentText()
        self._loading_edit = True
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItems(SPECIES_CATEGORY_ORDER)
        i = self.category_combo.findText(cur)
        self.category_combo.setCurrentIndex(i if i >= 0 else 0)
        self.category_combo.blockSignals(False)
        self._on_category_changed(self.category_combo.currentText())
        self._loading_edit = False

    # ------------------------------------------------- conspecific sharks
    def _make_shark_logger(self):
        """Compact ♂/♀/? + Log control shown at the end of the Social bar."""
        w = QWidget()
        hl = QHBoxLayout(w)
        hl.setContentsMargins(12, 0, 0, 0)
        hl.setSpacing(4)
        lab = QLabel("Sharks:")
        lab.setObjectName("BarTitle")
        lab.setToolTip("Log the conspecific sharks present right now:\n"
                       "set ♂ / ♀ / ? counts, then press Log.\n"
                       "Log again whenever the group changes.")
        hl.addWidget(lab)
        self.shark_m_spin = QSpinBox(); self.shark_f_spin = QSpinBox()
        self.shark_u_spin = QSpinBox()
        for sp, sym in ((self.shark_m_spin, "♂"), (self.shark_f_spin, "♀"),
                        (self.shark_u_spin, "?")):
            sp.setRange(0, 99)
            sp.setToolTip(f"Number of {sym} sharks present")
            hl.addWidget(QLabel(sym))
            hl.addWidget(sp)
        log_btn = QToolButton()
        log_btn.setText("Log")
        log_btn.setProperty("segmented", True)
        log_btn.setToolTip("Record this shark group at the current frame")
        log_btn.clicked.connect(self._log_sharks)
        hl.addWidget(log_btn)
        return w

    @staticmethod
    def _shark_entry_text(e) -> str:
        parts = []
        if e.get("males"):
            parts.append(f"{e['males']}♂")
        if e.get("females"):
            parts.append(f"{e['females']}♀")
        if e.get("unknown"):
            parts.append(f"{e['unknown']}?")
        total = e.get("males", 0) + e.get("females", 0) + e.get("unknown", 0)
        return f"{' '.join(parts) or '0'}  ({total} shark{'s' if total != 1 else ''})"

    def _log_sharks(self):
        m = self.shark_m_spin.value()
        f = self.shark_f_spin.value()
        u = self.shark_u_spin.value()
        if m + f + u == 0:
            self.statusBar().showMessage(
                "Set the ♂/♀/? counts first, then press Log.")
            return
        self._push_undo()
        fi = self.current_frame_idx
        seg_s, seg_e, seg_lab = self.store.social_segment_at(fi)
        # one composition per social interaction: replace any entry already
        # inside this segment instead of duplicating
        self.store.shark_log = [e for e in self.store.shark_log
                                if not (seg_s <= e["frame"] <= seg_e)]
        entry = {"frame": fi, "males": m, "females": f, "unknown": u}
        self.store.shark_log.append(entry)
        self.store.shark_log.sort(key=lambda e: e["frame"])
        self._refresh_shark_log()
        if hasattr(self, "timeline"):
            self.timeline.update()
        for sp in (self.shark_m_spin, self.shark_f_spin, self.shark_u_spin):
            sp.setValue(0)
        t = self.timeline._format_time
        if seg_lab:
            self.statusBar().showMessage(
                f"Logged {self._shark_entry_text(entry)} for the whole "
                f"'{seg_lab}' interaction "
                f"[{t(seg_s / self.fps)}–{t(seg_e / self.fps)}].")
        else:
            self.statusBar().showMessage(
                f"Logged {self._shark_entry_text(entry)} at frame {fi} — "
                "tip: label the Social axis and the entry will cover the "
                "whole interaction.")

    def _refresh_shark_log(self):
        if not hasattr(self, "shark_log_list"):
            return
        self.shark_log_list.clear()
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        tf = self.timeline._format_time
        fps = max(1e-6, self.fps)
        for e in self.store.shark_log:
            s, en, lab = self.store.social_segment_at(e["frame"])
            if lab:
                span = f"[{tf(s / fps)}–{tf(en / fps)}] {lab}"
            else:
                span = f"[{tf(e['frame'] / fps)}]"
            text = f"{span} · {self._shark_entry_text(e)}"
            it = QListWidgetItem(text)
            it.setToolTip(text + "\nApplies to the whole social interaction. "
                          "Double-click to jump; Delete Entry removes.")
            it.setData(user_role, e)
            self.shark_log_list.addItem(it)

    def _delete_shark_entry(self):
        it = self.shark_log_list.currentItem()
        if it is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        e = it.data(user_role)
        if e is not None:
            self._push_undo()
            try:
                self.store.shark_log.remove(e)
            except ValueError:
                pass
            self._refresh_shark_log()
            if hasattr(self, "timeline"):
                self.timeline.update()

    def _on_shark_entry_activated(self, item):
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        e = item.data(user_role)
        if e is not None:
            self.seek_to(int(e["frame"]))

    # -------------------------------------------------------------- notes
    def _refresh_notes(self):
        self.notes_list.clear()
        for note in self.store.notes:
            t = self.timeline._format_time(note["frame"] / max(1e-6, self.fps))
            it = QListWidgetItem(f"[{t}]  {note['text']}")
            it.setToolTip(note["text"])
            user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
            it.setData(user_role, note)
            self.notes_list.addItem(it)
        if hasattr(self, "timeline"):
            self.timeline.update()   # keep comment markers in sync

    def _add_note(self):
        text = (self.note_edit.text() or "").strip()
        if not text:
            return
        self._push_undo()
        self.store.add_note(self.current_frame_idx, text)
        self.note_edit.clear()
        self._refresh_notes()
        self.statusBar().showMessage(f"Note added at frame {self.current_frame_idx}.")

    def _delete_note(self):
        it = self.notes_list.currentItem()
        if it is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        note = it.data(user_role)
        if note is not None:
            self._push_undo()
            self.store.remove_note(note)
            self._refresh_notes()

    def _on_note_activated(self, item):
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        note = item.data(user_role)
        if note is not None:
            self.seek_to(int(note["frame"]))

    def _on_video_note_changed(self):
        new = (self.video_note_edit.text() or "").strip()
        if new != self.store.video_note:
            self.store.video_note = new
            self._mark_dirty()

    def _sync_video_note(self):
        if hasattr(self, "video_note_edit"):
            self.video_note_edit.setText(self.store.video_note or "")

    def _on_category_changed(self, category: str):
        """Narrow the taxonomy picker to the chosen animal category."""
        if hasattr(self, "taxon"):
            self.taxon.set_category(category)

    def _ensure_scene_labels(self, idx: int):
        stored_mov = self.store.movement_per_frame[idx]
        stored_soc = self.store.social_per_frame[idx]
        stored_hab = self.store.habitat_per_frame[idx]
        stored_vis = self.store.visibility_per_frame[idx]

        if not self.playing:
            # The bars are the *pen*, not a readout of the frame. Adopt a stored
            # label when the frame has one (so you can see and continue it), but
            # keep the current pen over unlabeled stretches so you can keep
            # labeling forward. Selecting "None" sets the pen to erase.
            if stored_mov is not None:
                self._current_movement = stored_mov
            if stored_soc is not None:
                self._current_social = stored_soc
            if stored_hab is not None:
                self._current_habitat = stored_hab
            if stored_vis is not None:
                self._current_visibility = stored_vis
        # During playback: _current_* stays fixed; _tick writes it to each frame.

        for cur, btns in ((self._current_movement, self._movement_btns),
                          (self._current_social, self._social_btns),
                          (self._current_habitat, self._habitat_btns),
                          (self._current_visibility, self._visibility_btns)):
            b = btns.get(cur)      # cur may be None -> the "None" button
            if b is not None:
                b.setChecked(True)

    # ------------------------------------------------------- feature list
    @staticmethod
    def _feature_row_text(feat) -> str:
        label = feat.species_category
        _idl = feat.id_label()
        if _idl and _idl != feat.species_category:
            label = f"{feat.species_category}: {_idl}"
        extra = ""
        summary = feat.sex_summary()
        if summary:
            extra += f"  [{summary}]"
        elif feat.sex:
            extra += {"Male": " ♂", "Female": " ♀", "Unknown": " ?"}.get(feat.sex, "")
        if feat.confidence:
            extra += f"  conf:{feat.confidence}"
        return f"{label}{extra}  ×{feat.count}  [f{feat.init_frame}]  {feat.name}"

    def _refresh_features(self):
        """Rebuild the box list, KEEPING the current selection.

        Signals are blocked so rebuilding never looks like the user picking
        (or un-picking) a box; selection is owned by _selected_feature_idx.
        """
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        self.feat_list.blockSignals(True)
        try:
            self.feat_list.clear()
            for i, feat in enumerate(self.store.features):
                c = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)]
                display = self._feature_row_text(feat)
                it = QListWidgetItem(display)
                it.setToolTip(display)   # full text on hover (rows elide when narrow)
                it.setData(user_role, i)
                it.setForeground(QColor(*c))
                self.feat_list.addItem(it)
            sel = self._selected_feature_idx
            if sel is not None and 0 <= sel < len(self.store.features):
                self.feat_list.setCurrentRow(sel)
                if getattr(self, "_new_box_fields", None) is None:
                    self._new_box_fields = self._fields_snapshot()
                self._populate_fields(self.store.features[sel])
            else:
                self.feat_list.setCurrentRow(-1)
        finally:
            self.feat_list.blockSignals(False)
        sel = self._selected_feature_idx
        if sel is not None and not (0 <= sel < len(self.store.features)):
            self._end_edit()
        elif sel is None and getattr(self, "_new_box_fields", None) is not None:
            self._end_edit()
        self._update_edit_banner()

    def _delete_feature(self):
        it = self.feat_list.currentItem()
        idx = None
        if it is not None:
            user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
            idx = it.data(user_role)
        if idx is None:
            idx = self._selected_feature_idx  # Delete key with no list focus
        if idx is None or not (0 <= int(idx) < len(self.store.features)):
            self.statusBar().showMessage(
                "Select a box first (click it on the video or in the list).")
            return
        self._push_undo()
        name = self.store.features[int(idx)].name
        self.store.remove_feature(int(idx))
        self._selected_feature_idx = None
        self._end_edit()
        self._refresh_features()
        self._display_frame(self.current_frame_idx)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Deleted '{name}'.  Press ⌘Z / Ctrl+Z to undo.")

    # -- selecting & editing a placed box ---------------------------------
    def _fields_snapshot(self) -> dict:
        return {"name": self.name_edit.text(),
                "category": self.category_combo.currentText(),
                "tax": self.taxon.value(),
                "count": self.count_spin.value(),
                "conf": self.confidence_combo.currentText()}

    def _set_fields(self, name, category, tax, count, conf):
        self._loading_edit = True
        try:
            self.name_edit.setText(name)
            ci = self.category_combo.findText(category)
            if ci >= 0:
                self.category_combo.setCurrentIndex(ci)
            self.taxon.set_value(tax.get("family", ""), tax.get("genus", ""),
                                 tax.get("species", ""), tax.get("common", ""))
            self.count_spin.setValue(max(1, int(count or 1)))
            ci = self.confidence_combo.findText(conf or "")
            self.confidence_combo.setCurrentIndex(ci if ci >= 0 else 0)
        finally:
            self._loading_edit = False

    def _populate_fields(self, feat):
        self._set_fields(feat.name, feat.species_category,
                         {"family": feat.family, "genus": feat.genus,
                          "species": feat.species, "common": feat.common},
                         feat.count, feat.confidence)

    def _select_feature(self, idx: int):
        """Enter edit mode on box `idx`: jump to a frame where it is drawn,
        show its corner handles, and load its details into the fields."""
        if not (0 <= idx < len(self.store.features)):
            return
        if self.bbox_btn.isChecked():
            self.bbox_btn.setChecked(False)
        if self._selected_feature_idx is None and self._new_box_fields is None:
            self._new_box_fields = self._fields_snapshot()
        self._selected_feature_idx = idx
        self._edit_undo_for = None
        frames = self.store.feature_bboxes[idx]
        if frames and self.current_frame_idx not in frames:
            # Boxes live on the single frame they were drawn on; bring that
            # frame up so the box is actually visible and editable.
            if self.playing:
                self.toggle_play()
            self.seek_to(min(frames, key=lambda f: abs(f - self.current_frame_idx)))
        self.feat_list.blockSignals(True)
        self.feat_list.setCurrentRow(idx)
        self.feat_list.blockSignals(False)
        self._populate_fields(self.store.features[idx])
        self._display_frame(self.current_frame_idx)
        self._update_edit_banner()
        self.statusBar().showMessage(
            f"Editing '{self.store.features[idx].name}' — changes apply "
            "immediately; drag a corner to resize; Esc when done.")

    def _end_edit(self):
        """Leave edit mode; the fields return to the next-box settings."""
        had = self._selected_feature_idx is not None
        self._selected_feature_idx = None
        self._edit_undo_for = None
        self.video_label.set_edit_rect(None)
        self.feat_list.blockSignals(True)
        self.feat_list.setCurrentRow(-1)
        self.feat_list.clearSelection()
        self.feat_list.blockSignals(False)
        stash = getattr(self, "_new_box_fields", None)
        self._new_box_fields = None
        if stash is not None:
            self._set_fields(stash["name"], stash["category"], stash["tax"],
                             stash["count"], stash["conf"])
        self._update_edit_banner()
        if had:
            self._display_frame(self.current_frame_idx)

    def _update_edit_banner(self):
        if not hasattr(self, "edit_banner"):
            return
        fi = self._selected_feature_idx
        if fi is not None and 0 <= fi < len(self.store.features):
            feat = self.store.features[fi]
            self.edit_banner.setText(
                f"<b>Editing box “{feat.name}”</b> (frame {feat.init_frame}). "
                "Changes below save instantly. Press <b>Esc</b> or "
                "<b>Done editing</b> when finished.")
            self.edit_banner.setStyleSheet(
                "QLabel { background:#FEF3C7; color:#78350F; border:1px solid #F59E0B;"
                " border-radius:6px; padding:5px; }")
            self.done_edit_btn.setEnabled(True)
        else:
            self.edit_banner.setText(
                "<b>Next box:</b> set the ID below, press <b>Draw Bbox</b> (B), "
                "then click two corners. Click any existing box to edit it.")
            self.edit_banner.setStyleSheet(
                "QLabel { background:#EEF2FF; color:#3730A3; border-radius:6px;"
                " padding:5px; }")
            self.done_edit_btn.setEnabled(False)

    def _apply_fields_to(self, feat):
        name = (self.name_edit.text() or "").strip()
        if name:
            feat.name = name
        feat.species_category = self.category_combo.currentText()
        _tax = self.taxon.value()
        feat.species = _tax["species"]
        feat.family = _tax["family"]
        feat.genus = _tax["genus"]
        feat.common = _tax["common"]
        feat.confidence = self.confidence_combo.currentText().strip()
        feat.count = self.count_spin.value()

    def _live_edit(self):
        """A field changed while a box is selected → write it to that box."""
        if getattr(self, "_loading_edit", True):
            return
        fi = self._selected_feature_idx
        if fi is None or not (0 <= fi < len(self.store.features)):
            return
        if self._edit_undo_for != fi:
            self._push_undo()          # one undo step per editing session
            self._edit_undo_for = fi
        else:
            self._mark_dirty()
        feat = self.store.features[fi]
        self._apply_fields_to(feat)
        it = self.feat_list.item(fi)
        if it is not None:
            txt = self._feature_row_text(feat)
            it.setText(txt)
            it.setToolTip(txt)
        self._display_frame(self.current_frame_idx)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self._update_edit_banner()

    def _on_feature_selected(self, current, previous):
        """User picked a row in the box list."""
        if current is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        idx = current.data(user_role)
        if idx is None or not (0 <= int(idx) < len(self.store.features)):
            return
        self._select_feature(int(idx))

    def _update_feature(self):
        """Explicitly write the fields to the selected box (kept for scripts;
        the UI now applies edits live)."""
        fi = self._selected_feature_idx
        if fi is None or not (0 <= fi < len(self.store.features)):
            return
        self._push_undo()
        feat = self.store.features[fi]
        self._apply_fields_to(feat)
        self._refresh_features()
        self._display_frame(self.current_frame_idx)
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Updated '{feat.name}'.")

    def _load_json(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Load annotation JSON", "", "JSON (*.json)")
        if not p:
            return
        try:
            self._push_undo()
            self._apply_loaded_store(FeatureStore.load_json(p))
            self.statusBar().showMessage(f"Loaded {p}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    # ------------------------------------------------------------- playback
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            interval = max(1, int(33 / self._play_speed))
            self.timer.start(interval)
        else:
            self.timer.stop()

    def _toggle_label_mode(self, checked: bool):
        self._labeling = checked
        self.label_mode_btn.setText("Labeling: ON" if checked else "Labeling: OFF")

    def _on_axis_toggle(self, axis: str, on: bool):
        self._label_axes[axis] = on
        active = [k for k, v in self._label_axes.items() if v]
        self.statusBar().showMessage(
            "Labeling axes: " + (", ".join(active) if active else "none"))

    def _on_speed_changed(self, text: str):
        self._play_speed = float(text.replace('x', ''))
        if self.timer.isActive():
            interval = max(1, int(33 / self._play_speed))
            self.timer.setInterval(interval)

    def _write_labels_at(self, f: int):
        """Write the current labels to frame f, honouring the per-axis toggles."""
        ax = self._label_axes
        if ax.get("movement", True):
            self.store.set_movement(f, self._current_movement)
        if ax.get("social", True):
            self.store.set_social(f, self._current_social)
        if ax.get("habitat", True):
            self.store.set_habitat(f, self._current_habitat)
        if ax.get("visibility", True):
            self.store.set_visibility(f, self._current_visibility)
        # labeling during playback/scrub doesn't go through _push_undo
        self._mark_dirty()

    def _on_slider_seek(self, idx: int):
        if self._labeling and idx > self.current_frame_idx:
            for f in range(self.current_frame_idx, idx + 1):
                self._write_labels_at(f)
            if hasattr(self, "timeline"):
                self.timeline.update()
            self._refresh_shark_log()   # social spans may have grown
        self.seek_to(idx)

    def _tick(self):
        if self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            self.play_btn.setText("Play")
            self.timer.stop()
            return
        if self._labeling:
            self._write_labels_at(self.current_frame_idx)
        self.seek_to(self.current_frame_idx + 1)

    def seek_to(self, idx: int):
        idx = max(0, min(int(idx), self.total_frames - 1))
        self.current_frame_idx = idx
        self._ensure_scene_labels(idx)
        self._display_frame(idx)

        if hasattr(self, "timeline"):
            self.timeline.set_current_frame(idx)

        # Keep this compact — it sits in the bottom control strip, and a long
        # string here used to force the whole window wider than small screens.
        _n = lambda v: v if v else "—"
        full = (f"Frame {idx}/{self.total_frames - 1} • {_n(self._current_movement)}"
                f" + {_n(self._current_social)} • {_n(self._current_habitat)}"
                f" • vis: {_n(self._current_visibility)}")
        self.frame_lbl.setText(f"Frame {idx}/{self.total_frames - 1}")
        self.frame_lbl.setToolTip(full)

    # -------------------------------------------------------------- display
    def _read_frame(self, idx):
        if self.is_frame_dir:
            if 0 <= idx < len(self.frame_files):
                return cv2.imread(self.frame_files[idx])
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        return frame if ok else None

    def _view_rect(self):
        """Compute the region of the full-res frame to show given zoom/center."""
        z = max(1.0, float(self._zoom))
        vw = int(round(self.video_w / z))
        vh = int(round(self.video_h / z))
        vw = max(1, min(vw, self.video_w))
        vh = max(1, min(vh, self.video_h))
        vx = int(round(self._zoom_cx - vw / 2))
        vy = int(round(self._zoom_cy - vh / 2))
        vx = max(0, min(vx, self.video_w - vw))
        vy = max(0, min(vy, self.video_h - vh))
        return vx, vy, vw, vh

    def _display_frame(self, idx):
        frame = self._read_frame(idx)
        if frame is None:
            return
        preview = frame.copy()

        # overlay committed features (bboxes only — no SAM2 masks)
        for _, feat, mask, bbox in self.store.features_at(idx):
            bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
            self._draw_mask(preview, mask, bgr, self._feature_label(feat), bbox)

        # boxes drawn on nearby frames: dashed, with how far away they are
        fps = max(1e-6, self.fps)
        for i, f, bbox in self._nearby_boxes(idx):
            feat = self.store.features[i]
            bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
            self._draw_dashed_rect(preview, bbox, bgr)
            dt = (f - idx) / fps
            tag = f"{feat.name} at {'+' if dt > 0 else ''}{dt:.1f}s"
            x1, y1 = int(bbox[0]), int(bbox[1])
            cv2.putText(preview, tag, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, tag, (x1, max(14, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)

        # scene text (anti-aliased)
        self._draw_scene_overlay(preview, idx)

        # crop to the zoom viewport
        vx, vy, vw, vh = self._view_rect()
        view = preview[vy:vy + vh, vx:vx + vw]

        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w,
                      QImage.Format.Format_RGB888 if _QT6 else QImage.Format_RGB888)
        aspect_mode = Qt.AspectRatioMode.KeepAspectRatio if _QT6 else Qt.KeepAspectRatio
        transform_mode = Qt.TransformationMode.SmoothTransformation if _QT6 else Qt.SmoothTransformation
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), aspect_mode, transform_mode)
        self.video_label.set_frame_pixmap(pix)

        # record the view transform for coord mapping
        lw, lh = self.video_label.width(), self.video_label.height()
        pw, ph = pix.width(), pix.height()
        ox, oy = (lw - pw) // 2, (lh - ph) // 2
        self._view = (vx, vy, vw, vh, pw, ph, ox, oy)

        # keep the selected box's edit handles in sync with the view
        self._update_edit_rect(idx)

    @staticmethod
    def _feature_label(feat) -> str:
        """Compose the on-frame label for a feature (species, sex, count, conf)."""
        lbl = feat.species_category
        _idl = feat.id_label()
        if _idl and _idl != feat.species_category:
            lbl = f"{feat.species_category}: {_idl}"
        summary = feat.sex_summary()
        if summary:
            lbl = f"{lbl} [{summary}]"
        elif feat.sex:
            sex_abbr = {"Male": "♂", "Female": "♀", "Unknown": "?"}.get(feat.sex, "")
            if sex_abbr:
                lbl = f"{lbl} {sex_abbr}"
        if feat.count > 1 and not summary:
            lbl = f"{lbl} x{feat.count}"
        if feat.confidence:
            lbl = f"{lbl} ({feat.confidence[:1].lower()})"
        return lbl

    def _draw_scene_overlay(self, img, idx):
        """Burn the current scene labels (movement/social/habitat/visibility) in."""
        mov = self.store.movement_per_frame[idx] or self._current_movement
        soc = self.store.social_per_frame[idx] or self._current_social
        hab = self.store.habitat_per_frame[idx] or self._current_habitat
        vis = self.store.visibility_per_frame[idx] or self._current_visibility
        line1 = f"{mov or '—'}  |  {soc or '—'}"
        line2 = f"{hab or '—'}  |  vis: {vis or '—'}"
        for i, text in enumerate((line1, line2)):
            y = 30 + i * 28
            cv2.putText(img, text, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_mask(img, mask, bgr_color, label, bbox):
        if mask is not None and mask.any():
            bgr = np.array(bgr_color, dtype=np.float64)
            img[mask] = (bgr * 0.35 + img[mask].astype(np.float64) * 0.65).astype(np.uint8)
            mu8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, bgr_color, 2, lineType=cv2.LINE_AA)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, 2, lineType=cv2.LINE_AA)
            if label:
                cv2.putText(img, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2, cv2.LINE_AA)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._display_frame(self.current_frame_idx)
        if hasattr(self, "timeline"):
            self.timeline.update()

    # ----------------------------------------------------------- zoom
    def _selected_bbox_on_frame(self, idx):
        """Return (feat_idx, bbox) if the selected feature has a box on `idx`."""
        fi = self._selected_feature_idx
        if fi is None or not (0 <= fi < len(self.store.features)):
            return None
        bbox = self.store.feature_bboxes[fi].get(idx)
        if bbox is None:
            return None
        return fi, bbox

    def _update_edit_rect(self, idx):
        """Push the selected feature's box to the VideoLabel as editable handles."""
        if not hasattr(self, "video_label"):
            return
        sel = self._selected_bbox_on_frame(idx)
        if sel is None or self.video_label._bbox_mode:
            self.video_label.set_edit_rect(None)
            return
        _, (x1, y1, x2, y2) = sel
        lx1, ly1 = self._video_to_label(x1, y1)
        lx2, ly2 = self._video_to_label(x2, y2)
        self.video_label.set_edit_rect([lx1, ly1, lx2, ly2])

    def _on_bbox_edited(self, lx1, ly1, lx2, ly2):
        """A corner handle was dragged — map back to video coords and store it."""
        sel = self._selected_bbox_on_frame(self.current_frame_idx)
        if sel is None:
            return
        fi, _ = sel
        m1 = self._map(lx1, ly1)
        m2 = self._map(lx2, ly2)
        if m1 is None or m2 is None:
            self._display_frame(self.current_frame_idx)  # snap back
            return
        x1, y1 = min(m1[0], m2[0]), min(m1[1], m2[1])
        x2, y2 = max(m1[0], m2[0]), max(m1[1], m2[1])
        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            self._display_frame(self.current_frame_idx)
            return
        self._push_undo()
        self.store.feature_bboxes[fi][self.current_frame_idx] = (x1, y1, x2, y2)
        feat = self.store.features[fi]
        if self.current_frame_idx == feat.init_frame:
            feat.init_coords = [x1, y1, x2, y2]
        self._display_frame(self.current_frame_idx)
        self.statusBar().showMessage(f"Resized box for '{feat.name}'.")

    def _on_video_click(self, pos):
        """Click on the video (not drawing/editing): select the box under the
        cursor; a dashed box from a nearby frame jumps there; empty space
        leaves edit mode."""
        m = self._map(pos.x(), pos.y())
        if m is None:
            return
        px, py = m
        def inside(b):
            x1, y1, x2, y2 = b
            return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)
        hit = None
        for i, feat, mask, bbox in self.store.features_at(self.current_frame_idx):
            if bbox is not None and inside(bbox):
                hit = i  # last (topmost) match wins
        if hit is None:
            for i, f, bbox in self._nearby_boxes(self.current_frame_idx):
                if inside(bbox):
                    hit = i
                    break
        if hit is not None:
            self._select_feature(hit)
        elif self._selected_feature_idx is not None:
            self._end_edit()

    def _nearby_boxes(self, idx, radius=None):
        """(feat_idx, frame, bbox) for boxes drawn within `radius` frames of
        idx but not on idx itself — shown dashed so a box never seems to have
        vanished just because you came back one frame off."""
        if radius is None:
            radius = max(1, int(round(self.fps)))      # ±1 second
        out = []
        for i, frames in enumerate(self.store.feature_bboxes):
            if not frames or idx in frames:
                continue
            f = min(frames, key=lambda k: abs(k - idx))
            if abs(f - idx) <= radius:
                out.append((i, f, frames[f]))
        return out

    @staticmethod
    def _draw_dashed_rect(img, bbox, bgr, dash=10, thick=1):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        def seg(ax, ay, bx, by):
            length = max(abs(bx - ax), abs(by - ay))
            for s0 in range(0, length, dash * 2):
                s1 = min(length, s0 + dash)
                t0, t1 = s0 / max(1, length), s1 / max(1, length)
                cv2.line(img, (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0)),
                         (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1)),
                         bgr, thick, cv2.LINE_AA)
        seg(x1, y1, x2, y1); seg(x2, y1, x2, y2)
        seg(x2, y2, x1, y2); seg(x1, y2, x1, y1)

    def _toggle_draw_shortcut(self):
        self.bbox_btn.setChecked(not self.bbox_btn.isChecked())

    def _focus_comment(self):
        self.note_edit.setFocus()

    def keyPressEvent(self, ev):
        """Bare-key shortcuts + Delete, but only when a text field isn't focused
        (so typing spaces/letters/backspace in a field is never hijacked)."""
        fw = QApplication.focusWidget()
        typing = isinstance(fw, (QLineEdit, QComboBox, QSpinBox))
        key = ev.key()
        K = Qt.Key if _QT6 else Qt

        if key == K.Key_Escape:
            if self.bbox_btn.isChecked():
                self.bbox_btn.setChecked(False)
            elif self._selected_feature_idx is not None:
                self._end_edit()
            ev.accept()
            return
        if not typing:
            if key in ((K.Key_Delete, K.Key_Backspace)):
                # only delete a box that is visibly selected on this frame, so a
                # stray Backspace can't remove something you can't see
                if self._selected_bbox_on_frame(self.current_frame_idx) is not None:
                    self._delete_feature()
                    ev.accept()
                    return
            elif key == K.Key_Space:
                self.toggle_play(); ev.accept(); return
            elif key == K.Key_B:
                self._toggle_draw_shortcut(); ev.accept(); return
            elif key == K.Key_C:
                self._focus_comment(); ev.accept(); return
            elif key == K.Key_F:
                self._toggle_focus_mode(); ev.accept(); return
            elif key == K.Key_I:
                self._mark_in(); ev.accept(); return
            elif key == K.Key_O:
                self._mark_out(); ev.accept(); return
        super().keyPressEvent(ev)

    def _zoom_at(self, factor, anchor_label=None):
        """Multiply zoom by `factor`, keeping the point under `anchor_label` fixed."""
        old = self._zoom
        new = max(1.0, min(8.0, old * factor))
        if abs(new - old) < 1e-3:
            return
        # anchor: recentre so the cursor's video point stays put
        if anchor_label is not None and self._view:
            m = self._map(anchor_label.x(), anchor_label.y())
            if m is not None:
                ax, ay = m
                self._zoom = new
                # new view centred to keep (ax,ay) near the same spot
                self._zoom_cx = ax
                self._zoom_cy = ay
            else:
                self._zoom = new
        else:
            self._zoom = new
        if new <= 1.0:
            self._zoom_cx = self.video_w / 2.0
            self._zoom_cy = self.video_h / 2.0
        self._display_frame(self.current_frame_idx)
        self._update_zoom_label()

    def _on_wheel_zoom(self, delta, anchor):
        self._zoom_at(1.15 if delta > 0 else 1 / 1.15, anchor)

    def _zoom_in(self):
        self._zoom_at(1.25, None)

    def _zoom_out(self):
        self._zoom_at(1 / 1.25, None)

    def _zoom_reset(self):
        self._zoom = 1.0
        self._zoom_cx = self.video_w / 2.0
        self._zoom_cy = self.video_h / 2.0
        self._display_frame(self.current_frame_idx)
        self._update_zoom_label()

    def _update_zoom_label(self):
        if hasattr(self, "zoom_lbl"):
            self.zoom_lbl.setText(f"{self._zoom:.1f}×")

    # ------------------------------------------------------------- undo
    def _snapshot(self) -> dict:
        s = self.store
        return {
            "movement": list(s.movement_per_frame),
            "social": list(s.social_per_frame),
            "habitat": list(s.habitat_per_frame),
            "visibility": list(s.visibility_per_frame),
            "features": copy.deepcopy(s.features),
            "bboxes": copy.deepcopy(s.feature_bboxes),
            "masks": [dict(m) for m in s.feature_masks],
            "next_color": s._next_color,
            "notes": copy.deepcopy(s.notes),
            "clips": copy.deepcopy(s.clips),
            "video_note": s.video_note,
            "shark_log": copy.deepcopy(s.shark_log),
            "sel": self._selected_feature_idx,
        }

    def _push_undo(self):
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self._undo_limit:
            self._undo_stack.pop(0)
        # every discrete mutating action calls _push_undo first, so this is the
        # single choke point that marks the session dirty
        self._mark_dirty()

    def _undo(self):
        if not self._undo_stack:
            self.statusBar().showMessage("Nothing to undo.")
            return
        snap = self._undo_stack.pop()
        s = self.store
        s.movement_per_frame = snap["movement"]
        s.social_per_frame = snap["social"]
        s.habitat_per_frame = snap["habitat"]
        s.visibility_per_frame = snap["visibility"]
        s.features = snap["features"]
        s.feature_bboxes = snap["bboxes"]
        s.feature_masks = snap["masks"]
        s._next_color = snap["next_color"]
        s.notes = snap["notes"]
        s.clips = snap["clips"]
        s.video_note = snap.get("video_note", "")
        s.shark_log = snap.get("shark_log", [])
        self._selected_feature_idx = snap["sel"]
        self._edit_undo_for = None     # next live edit gets its own undo step
        self._refresh_features()
        self._refresh_notes()
        self._refresh_clips()
        self._refresh_shark_log()
        self._sync_video_note()
        if hasattr(self, "timeline"):
            self.timeline.update()
        self._display_frame(self.current_frame_idx)
        self._mark_dirty()   # undo is itself a change that must be persisted
        self.statusBar().showMessage("Undo.")

    # ------------------------------------------------------- clips
    @staticmethod
    def _std_buttons():
        if _QT6:
            return (QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No)
        return (QMessageBox.Yes, QMessageBox.No)

    def _mark_in(self):
        self._clip_in = self.current_frame_idx
        if self._clip_out is not None and self._clip_out < self._clip_in:
            self._clip_out = None
        self._update_clip_label()

    def _mark_out(self):
        self._clip_out = self.current_frame_idx
        if self._clip_in is not None and self._clip_in > self._clip_out:
            self._clip_in = None
        self._update_clip_label()

    def _update_clip_label(self):
        def t(f):
            return (self.timeline._format_time(f / max(1e-6, self.fps))
                    if f is not None else "—")
        if hasattr(self, "clip_range_lbl"):
            self.clip_range_lbl.setText(
                f"In {t(self._clip_in)}  ·  Out {t(self._clip_out)}")
        if hasattr(self, "timeline"):
            self.timeline.clip_in = self._clip_in
            self.timeline.clip_out = self._clip_out
            self.timeline.update()

    def _save_clip(self):
        if self._clip_in is None or self._clip_out is None:
            QMessageBox.information(self, "Clip", "Set both In and Out marks first.")
            return
        a, b = sorted((self._clip_in, self._clip_out))
        try:
            from PyQt6.QtWidgets import QInputDialog
        except ImportError:
            from PyQt5.QtWidgets import QInputDialog
        default = f"clip_{len(self.store.clips) + 1}"
        name, ok = QInputDialog.getText(self, "Save clip", "Clip name:", text=default)
        if not ok:
            return
        name = (name or default).strip()
        self._push_undo()
        self.store.clips.append({"name": name, "start": a, "end": b})
        self._refresh_clips()
        self.statusBar().showMessage(f"Saved clip '{name}' [{a}–{b}].")

    def _refresh_clips(self):
        if not hasattr(self, "clips_list"):
            return
        self.clips_list.clear()
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        for i, c in enumerate(self.store.clips):
            t = self.timeline._format_time
            dur = (c["end"] - c["start"] + 1) / max(1e-6, self.fps)
            it = QListWidgetItem(
                f"{c['name']}  [{t(c['start'] / self.fps)}–{t(c['end'] / self.fps)}]"
                f"  {dur:.1f}s")
            it.setData(user_role, i)
            self.clips_list.addItem(it)

    def _delete_clip(self):
        it = self.clips_list.currentItem()
        if it is None:
            return
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        i = it.data(user_role)
        if i is not None and 0 <= i < len(self.store.clips):
            self._push_undo()
            self.store.clips.pop(i)
            self._refresh_clips()
            if hasattr(self, "timeline"):
                self.timeline.update()

    def _on_clip_activated(self, item):
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        i = item.data(user_role)
        if i is not None and 0 <= i < len(self.store.clips):
            c = self.store.clips[i]
            self._clip_in, self._clip_out = c["start"], c["end"]
            self._update_clip_label()
            self.seek_to(c["start"])

    def _export_clip(self):
        it = self.clips_list.currentItem() if hasattr(self, "clips_list") else None
        user_role = Qt.ItemDataRole.UserRole if _QT6 else Qt.UserRole
        if it is not None:
            i = it.data(user_role)
            c = self.store.clips[i]
            a, b, nm = c["start"], c["end"], c["name"]
        elif self._clip_in is not None and self._clip_out is not None:
            a, b = sorted((self._clip_in, self._clip_out))
            nm = "clip"
        else:
            QMessageBox.information(
                self, "Export clip",
                "Select a saved clip in the list, or set In/Out marks first.")
            return
        if self.playing:
            self.toggle_play()
        yes, no = self._std_buttons()
        ret = QMessageBox.question(
            self, "Annotation overlays",
            "Burn annotation overlays (labels + boxes) into the exported clip?",
            yes | no)
        overlays = (ret == yes)
        p, _ = QFileDialog.getSaveFileName(
            self, "Export clip", f"{nm}.mp4", "MP4 (*.mp4)")
        if not p:
            return
        try:
            self._render_clip(p, a, b, overlays)
            QMessageBox.information(self, "Done", f"Clip exported:\n{p}")
        except Exception as e:
            QMessageBox.critical(self, "Clip export error", str(e))

    def _render_clip(self, out_path, start, end, overlays):
        start = max(0, min(int(start), self.total_frames - 1))
        end = max(0, min(int(end), self.total_frames - 1))
        if end < start:
            start, end = end, start
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(out_path, fourcc, self.fps, (self.video_w, self.video_h))
        for fidx in range(start, end + 1):
            frame = self._read_frame(fidx)
            if frame is None:
                continue
            if overlays:
                for _, feat, mask, bbox in self.store.features_at(fidx):
                    bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
                    self._draw_mask(frame, mask, bgr, self._feature_label(feat), bbox)
                self._draw_scene_overlay(frame, fidx)
            w.write(frame)
        w.release()

    # -------------------------------------------------------- autosave
    def _source_key(self) -> str:
        """A filesystem-safe, stable id for the current source (for local autosave)."""
        return source_key_for(self.store.video_path or self.video_path or "session")

    def _autosave_path(self) -> str:
        """Autosave always goes to the guaranteed-local annotations folder, so it
        works even when the video is streamed from a cloud mount."""
        os.makedirs(LOCAL_ANNOTATIONS_DIR, exist_ok=True)
        return os.path.join(LOCAL_ANNOTATIONS_DIR,
                            self._source_key() + ".autosave.json")

    def _legacy_sidecar_path(self) -> str:
        """Old autosave location (next to the video) — checked for resume only."""
        base = self.store.video_path or self.video_path
        if base and os.path.isdir(base):
            return os.path.join(base, ".ctag_autosave.json")
        return (base or self.video_path) + ".ctag_autosave.json"

    def _safe_save(self, path: str):
        """Write the store to `path` and confirm it materialised.

        Returns (ok, error_message).  Catches cloud-mount write failures that
        would otherwise be silent, and verifies a non-empty file exists after.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            self.store.save_json(path)
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False, ("File did not appear on disk after writing "
                               "(the folder may be a cloud mount that rejected "
                               "the write).")
            return True, ""
        except Exception as e:
            return False, str(e)

    def _has_annotations(self) -> bool:
        """True if the store holds ANY annotation worth persisting.

        Must cover every kind of user work — missing one here means that work
        silently never autosaves (this previously omitted shark_log and
        video_note, so shark-only sessions saved nothing).
        """
        s = self.store
        if s.features or s.notes or s.clips or s.shark_log:
            return True
        if (s.video_note or "").strip():
            return True
        for tl in (s.movement_per_frame, s.social_per_frame,
                   s.habitat_per_frame, s.visibility_per_frame):
            if any(v is not None for v in tl):
                return True
        return False

    def _mark_dirty(self):
        """Record that something changed and schedule a prompt autosave.

        Called from every mutating action, so work is persisted within a couple
        of seconds of any edit rather than only on the 60 s tick / on close.
        """
        self._dirty = True
        if hasattr(self, "_dirty_timer"):
            self._dirty_timer.start(self._dirty_delay_ms)  # restart = debounce

    def _autosave(self, reason="timer"):
        if hasattr(self, "_dirty_timer"):
            self._dirty_timer.stop()
        if not self._has_annotations():
            return
        ok, err = self._safe_save(self._autosave_path())
        if ok:
            self._dirty = False
            self.statusBar().showMessage("Autosaved (local).", 2000)
        else:
            self.statusBar().showMessage(f"Autosave failed: {err}", 6000)

    def _apply_loaded_store(self, loaded: "FeatureStore"):
        """Merge a loaded store's annotations into the current store."""
        n = min(loaded.total_frames, self.store.total_frames)
        for fi in range(n):
            for src, dst in (
                (loaded.movement_per_frame, self.store.movement_per_frame),
                (loaded.social_per_frame, self.store.social_per_frame),
                (loaded.habitat_per_frame, self.store.habitat_per_frame),
                (loaded.visibility_per_frame, self.store.visibility_per_frame),
            ):
                if fi < len(src) and src[fi] is not None:
                    dst[fi] = src[fi]
        for feat, masks, bboxes in zip(loaded.features, loaded.feature_masks,
                                       loaded.feature_bboxes):
            self.store.features.append(feat)
            self.store.feature_masks.append(masks)
            self.store.feature_bboxes.append(bboxes)
        for note in loaded.notes:
            self.store.notes.append(note)
        self.store.notes.sort(key=lambda nn: nn["frame"])
        for clip in loaded.clips:
            self.store.clips.append(clip)
        for e in loaded.shark_log:
            self.store.shark_log.append(e)
        self.store.shark_log.sort(key=lambda e: e["frame"])
        if loaded.video_note and not self.store.video_note:
            self.store.video_note = loaded.video_note
        self._refresh_features()
        self._refresh_notes()
        self._refresh_clips()
        self._refresh_shark_log()
        self._sync_video_note()
        if hasattr(self, "timeline"):
            self.timeline.update()
        self._display_frame(self.current_frame_idx)

    def _maybe_resume_autosave(self):
        if self._has_annotations():
            return
        # Prefer the new local autosave; fall back to a legacy sidecar if present.
        p = self._autosave_path()
        if not os.path.exists(p):
            legacy = self._legacy_sidecar_path()
            p = legacy if os.path.exists(legacy) else None
        if not p:
            return
        yes, no = self._std_buttons()
        ret = QMessageBox.question(
            self, "Resume previous work",
            f"Found autosaved annotations for this video:\n{p}\n\nLoad them?",
            yes | no)
        if ret == yes:
            try:
                self._apply_loaded_store(FeatureStore.load_json(p))
                self.statusBar().showMessage("Resumed autosaved annotations.")
            except Exception as e:
                QMessageBox.warning(self, "Resume failed", str(e))

    # ------------------------------------------------ multi-video / folders
    def _open_video_dialog(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Video (*.mp4 *.MP4 *.mov *.avi *.mkv)")
        if p:
            self._cleanup_preloads()
            self._playlist = [p]
            self._refresh_playlist()
            self._switch_source(p, is_frame_dir=False)

    def _open_frames_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open frames directory")
        if d:
            self._cleanup_preloads()
            self._playlist = [d]
            self._refresh_playlist()
            self._switch_source(d, is_frame_dir=True)

    def _open_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Open folder of videos")
        if not d:
            return
        exts = (".mp4", ".mov", ".avi", ".mkv")
        vids = [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.lower().endswith(exts)]
        if not vids:
            QMessageBox.information(self, "Open folder",
                                    "No video files found in that folder.")
            return
        self._cleanup_preloads()
        self._playlist = vids
        self._refresh_playlist()
        self._switch_source(vids[0], is_frame_dir=False)

    def _refresh_playlist(self):
        """Rebuild the clip dropdown, ticking clips that already have work
        saved, and select whichever one is currently open."""
        if not hasattr(self, "video_combo"):
            return
        current = self.store.video_path or self.video_path
        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        sel = -1
        for i, p in enumerate(getattr(self, "_playlist", [])):
            mark = "✓ " if is_annotated(p) else "    "
            self.video_combo.addItem(mark + os.path.basename(p), p)
            self.video_combo.setItemData(
                i, ("Annotated — " if is_annotated(p) else "") + p,
                Qt.ItemDataRole.ToolTipRole if _QT6 else Qt.ToolTipRole)
            if p == current:
                sel = i
        if sel >= 0:
            self.video_combo.setCurrentIndex(sel)
        self.video_combo.blockSignals(False)

    def _on_playlist_changed(self, i):
        if i < 0 or i >= len(getattr(self, "_playlist", [])):
            return
        p = self._playlist[i]
        self._switch_source(p, is_frame_dir=os.path.isdir(p))

    # -- background preloading of the next playlist video ----------------
    def _start_preload_next(self):
        """Silently prepare the next playlist entry in a background thread."""
        playlist = getattr(self, "_playlist", [])
        if len(playlist) < 2 or self._preload_inflight is not None:
            return
        cur = self.store.video_path or self.video_path
        try:
            i = playlist.index(cur)
        except ValueError:
            i = -1
        nxt = playlist[i + 1] if 0 <= i + 1 < len(playlist) else None
        if not nxt or nxt in self._preload or nxt == cur:
            return
        self._preload_inflight = nxt

        def worker(p=nxt):
            try:
                prep = prepare_source(p, os.path.isdir(p), silent=True)
                self._preload[p] = prep
            except Exception:
                pass
            finally:
                self._preload_inflight = None

        threading.Thread(target=worker, daemon=True).start()
        self.statusBar().showMessage(
            f"Preloading next video in background: {os.path.basename(nxt)}", 4000)

    def _cleanup_preloads(self):
        """Delete temp frame dirs of preloads that were never used."""
        for prep in list(self._preload.values()):   # worker may still insert
            td = prep.get("temp_dir")
            if td and os.path.isdir(td):
                shutil.rmtree(td, ignore_errors=True)
        self._preload.clear()

    def _switch_source(self, path, is_frame_dir):
        # persist current work first
        self._autosave()
        if self.playing:
            self.toggle_play()
        prep = self._preload.pop(path, None)   # instant if preloaded
        if prep is None:
            try:
                prep = prepare_source(path, is_frame_dir)
            except Exception as e:
                QMessageBox.critical(self, "Open failed", str(e))
                return
        # tear down old source
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        old_temp = self.temp_dir
        # adopt new source
        self.video_path = prep["frames_source"]
        self.is_frame_dir = prep["is_frame_dir"]
        self.temp_dir = prep["temp_dir"]
        self.fps = prep["fps"] or 30.0
        self.total_frames = prep["total_frames"]
        self.video_w = prep["video_w"]
        self.video_h = prep["video_h"]
        if self.is_frame_dir:
            self.frame_files = prep["frame_files"]
            self.cap = None
        else:
            self.frame_files = None
            self.cap = cv2.VideoCapture(self.video_path)
        self.store = FeatureStore(prep["original"] or self.video_path, self.fps,
                                  self.total_frames, self.video_w, self.video_h)
        # reset transient state
        self._selected_feature_idx = None
        self._undo_stack.clear()
        self._clip_in = self._clip_out = None
        self._zoom = 1.0
        self._zoom_cx, self._zoom_cy = self.video_w / 2.0, self.video_h / 2.0
        # point the timeline at the new store
        self.timeline.store = self.store
        self.timeline.fps = self.fps
        self.timeline.total_frames = max(1, self.total_frames)
        self.timeline.current_frame = 0
        self.timeline.clip_in = self.timeline.clip_out = None
        self.setWindowTitle(f"CTAG Annotator — {os.path.basename(path)}")
        self.current_frame_idx = 0
        self._refresh_features()
        self._refresh_notes()
        self._refresh_clips()
        self._refresh_shark_log()
        self._sync_video_note()
        self._update_clip_label()
        self._update_zoom_label()
        self.seek_to(0)
        # clean up the previous temp frames dir (if any)
        if old_temp and os.path.isdir(old_temp):
            shutil.rmtree(old_temp, ignore_errors=True)
        self._maybe_resume_autosave()
        self._refresh_playlist()   # keep ✓ marks + selection in sync
        # start silently preparing the next playlist entry
        self._start_preload_next()

    # -------------------------------------------------------------- export
    def _save_plots(self, base_path: str) -> list:
        """Generate and save three analysis PNG plots alongside the exported video."""
        try:
            import matplotlib.pyplot as plt
            plt.switch_backend("agg")   # safe even if pyplot was already imported
            from collections import Counter, defaultdict
        except ImportError:
            QMessageBox.warning(self, "Plots skipped",
                                "matplotlib is not installed.\n"
                                "Run: pip install matplotlib")
            return []
        except Exception as e:
            QMessageBox.warning(self, "Plots skipped", f"Could not initialise matplotlib:\n{e}")
            return []

        try:
            return self._render_plots(base_path, plt, Counter, defaultdict)
        except Exception as e:
            QMessageBox.warning(self, "Plot error", f"Error generating plots:\n{e}")
            return []

    @staticmethod
    def _propagate(timeline):
        """Forward-fill a per-frame label list (None gaps inherit the last set value)."""
        out     = list(timeline)
        current = None
        for i, v in enumerate(out):
            if v is not None:
                current = v
            elif current is not None:
                out[i] = current
        return out

    def _render_plots(self, base_path, plt, Counter, defaultdict) -> list:
        from matplotlib.patches import Patch

        store = self.store
        total = self.total_frames
        saved = []

        # Forward-propagate so every frame inherits its nearest prior label.
        # This means a habitat annotated at frame 20 applies to frames 20-N
        # until the next annotation, matching the "persist until changed" UX.
        hab_timeline = self._propagate(store.habitat_per_frame)
        mov_timeline = self._propagate(store.movement_per_frame)
        soc_timeline = self._propagate(store.social_per_frame)
        vis_timeline = self._propagate(store.visibility_per_frame)
        # Movement is the primary "behavior" axis for the per-habitat plots.
        beh_timeline = mov_timeline

        WHITE = "#FFFFFF"
        BG    = "#F8FAFC"

        HAB_COLORS = {
            "Mangrove":     "#10B981",
            "Rocky reef":   "#0EA5E9",
            "Sandy bottom": "#F59E0B",
            "Gravel":       "#9CA3AF",
            "Mud":          "#78644A",
        }
        VIS_COLORS = {
            "Very bad":  "#BE123C",
            "Bad":       "#EF4444",
            "Fine":      "#F59E0B",
            "Good":      "#22C55E",
            "Very good": "#157F3D",
        }
        BEH_COLORS = [
            "#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444",
            "#EC4899", "#8B5CF6", "#0EA5E9", "#14B8A6",
        ]
        CAT_COLORS = {
            "Shark":        "#3B82F6",
            "Ray":          "#06B6D4",
            "Teleost fish": "#10B981",
            "Sea turtle":   "#F59E0B",
            "Other":        "#94A3B8",
        }
        DEFAULT_CLR = "#94A3B8"

        def _donut(ax, short_labels, vals, colors, title, title_color="#0F172A",
                   center_text=""):
            _, _, autotexts = ax.pie(
                vals, labels=None, colors=colors,
                autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
                pctdistance=0.78,
                startangle=90, counterclock=False,
                wedgeprops={"linewidth": 3, "edgecolor": WHITE, "width": 0.52},
            )
            for at in autotexts:
                at.set_fontsize(9); at.set_fontweight("bold"); at.set_color("#1E293B")
            ax.text(0, 0, center_text, ha="center", va="center",
                    fontsize=13, fontweight="bold", color="#0F172A", linespacing=1.5)
            ax.set_title(title, fontsize=16, fontweight="bold", pad=20,
                         color=title_color)

        def _hbar(ax, species, counts, colors, cat_for_sp, title,
                  title_color="#0F172A"):
            peak = max(counts)
            y    = np.arange(len(species))
            bars = ax.barh(y, counts, color=colors, height=0.58,
                           edgecolor=WHITE, linewidth=1.5, zorder=3)
            for bar, c in zip(bars, counts):
                ax.text(bar.get_width() + peak * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        str(c), va="center", ha="left",
                        fontsize=10, fontweight="700", color="#1E293B")
            ax.set_yticks(y)
            ax.set_yticklabels(species, fontsize=10, color="#334155")
            ax.set_xlabel("Animals Observed", fontsize=11, labelpad=8, color="#334155")
            ax.set_title(title, fontsize=16, fontweight="bold",
                         pad=14, color=title_color)
            ax.set_xlim(0, peak * 1.18)
            ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", colors="#475569", labelsize=10)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#CBD5E1")
            seen_cats = sorted({cat_for_sp.get(sp, "Other") for sp in species})
            if len(seen_cats) > 1:
                handles = [Patch(facecolor=CAT_COLORS.get(c, DEFAULT_CLR),
                                 edgecolor=WHITE, label=c) for c in seen_cats]
                leg = ax.legend(handles=handles, title="Category",
                                fontsize=9, title_fontsize=10,
                                framealpha=0.95, edgecolor="#E2E8F0", loc="lower right")
                leg.get_frame().set_linewidth(0.8)

        def _fish_badge(ax, species, counts_map, cat_for_sp):
            fish = [s for s in species if cat_for_sp.get(s) == "Teleost fish"]
            if not fish:
                return
            n_sp  = len(fish)
            total_fish = sum(counts_map[s] for s in fish)
            label = f"Teleost fish: {n_sp} {'species' if n_sp != 1 else 'species'}  ·  {total_fish} individuals"
            ax.text(0.01, 0.99, label, transform=ax.transAxes,
                    fontsize=9.5, va="top", ha="left",
                    color=CAT_COLORS["Teleost fish"], fontweight="600",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ECFDF5",
                              edgecolor="#6EE7B7", linewidth=1.2, alpha=0.92))

        def _feat_habitat(feat):
            # Use propagated timeline so a bbox at frame 5 picks up habitat
            # annotated at frame 1 even if frame 5 itself has no raw label.
            habs = [hab_timeline[f]
                    for f in range(feat.init_frame, min(feat.end_frame + 1, total))
                    if hab_timeline[f]]
            return Counter(habs).most_common(1)[0][0] if habs else None

        # ── build global species / cat maps and habitat list ─────────
        species_counts = Counter()
        cat_for_species = {}
        for feat in store.features:
            sp = feat.id_label() or "Unknown"
            species_counts[sp] += feat.count
            cat_for_species[sp] = feat.species_category or "Other"

        all_habitats = sorted(set(h for h in hab_timeline if h))

        # ── 1. Species observed — overall + per-habitat side by side ───
        if species_counts:
            # All species sorted by total count (ascending = top of chart = most)
            all_sp  = sorted(species_counts, key=lambda s: species_counts[s])
            n_sp    = len(all_sp)
            y_pos   = np.arange(n_sp)

            # Per-habitat species data (same _feat_habitat logic)
            sp_hab_counts = {}   # hab -> Counter
            sp_hab_cats   = {}   # hab -> {sp: cat}
            for hab in all_habitats:
                c, cat = Counter(), {}
                for feat in store.features:
                    if _feat_habitat(feat) == hab:
                        sp = feat.id_label() or "Unknown"
                        c[sp]   += feat.count
                        cat[sp]  = feat.species_category or "Other"
                sp_hab_counts[hab] = c
                sp_hab_cats[hab]   = cat

            n_cols  = 1 + len(all_habitats)
            height  = max(5.0, n_sp * 0.42 + 2.2)
            # First col needs room for species labels; rest just bars + count labels
            col_w   = [3.8] + [2.6] * len(all_habitats)
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(sum(col_w), height),
                sharey=True,
                gridspec_kw={"width_ratios": col_w},
            )
            if n_cols == 1:
                axes = [axes]
            fig.patch.set_facecolor(WHITE)
            fig.subplots_adjust(wspace=0.06)

            def _hbar_col(ax, counts_map, cat_map, title, title_color="#0F172A",
                          show_ylabels=True):
                counts = [counts_map.get(s, 0) for s in all_sp]
                colors = [CAT_COLORS.get(cat_map.get(s, "Other"), DEFAULT_CLR)
                          for s in all_sp]
                peak   = max((c for c in counts if c > 0), default=1)
                bars   = ax.barh(y_pos, counts, color=colors, height=0.62,
                                 edgecolor=WHITE, linewidth=1.2, zorder=3)
                for bar, c in zip(bars, counts):
                    if c > 0:
                        ax.text(bar.get_width() + peak * 0.04,
                                bar.get_y() + bar.get_height() / 2,
                                str(c), va="center", ha="left",
                                fontsize=8.5, fontweight="700", color="#1E293B")
                ax.set_yticks(y_pos)
                if show_ylabels:
                    ax.set_yticklabels(all_sp, fontsize=9, color="#334155")
                else:
                    ax.set_yticklabels([])
                ax.set_title(title, fontsize=11, fontweight="bold",
                             pad=10, color=title_color)
                ax.set_xlim(0, peak * 1.22)
                ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.7, zorder=0)
                ax.set_axisbelow(True)
                ax.set_facecolor(BG)
                ax.tick_params(axis="both", colors="#475569", labelsize=8.5)
                for sp in ("top", "right", "left"):
                    ax.spines[sp].set_visible(False)
                ax.spines["bottom"].set_color("#CBD5E1")
                ax.set_xlabel("Count", fontsize=9, labelpad=5, color="#334155")
                # Fish badge
                fish = [s for s in all_sp
                        if cat_map.get(s) == "Teleost fish"
                        and counts_map.get(s, 0) > 0]
                if fish:
                    n_f  = len(fish)
                    tot  = sum(counts_map[s] for s in fish)
                    lbl  = f"🐟 {n_f} sp · {tot} ind"
                    ax.text(0.97, 0.01, lbl, transform=ax.transAxes,
                            fontsize=8, va="bottom", ha="right",
                            color=CAT_COLORS["Teleost fish"], fontweight="600",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5",
                                      edgecolor="#6EE7B7", linewidth=1.0, alpha=0.9))

            # Overall column
            _hbar_col(axes[0], species_counts, cat_for_species,
                      "All Habitats", show_ylabels=True)

            # Per-habitat columns
            for i, hab in enumerate(all_habitats):
                _hbar_col(axes[i + 1],
                          sp_hab_counts[hab], sp_hab_cats[hab],
                          hab, title_color=HAB_COLORS.get(hab, DEFAULT_CLR),
                          show_ylabels=False)

            # Category legend on the overall column
            seen_cats = sorted({cat_for_species[s] for s in all_sp})
            if len(seen_cats) > 1:
                handles = [Patch(facecolor=CAT_COLORS.get(c, DEFAULT_CLR),
                                 edgecolor=WHITE, label=c) for c in seen_cats]
                leg = axes[0].legend(handles=handles, title="Category",
                                     fontsize=8, title_fontsize=9,
                                     framealpha=0.95, edgecolor="#E2E8F0",
                                     loc="lower right")
                leg.get_frame().set_linewidth(0.8)

            out = f"{base_path}_1_species.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
            plt.close(fig)
            saved.append(out)

        # ── 2. Behavior composition donuts — movement + social axes ───
        def _axis_donut(timeline, options, title, out_name):
            counts = Counter(lab for lab in timeline if lab)
            if not counts:
                return
            ordered = [b for b in options if b in counts]
            ordered += [b for b in counts if b not in ordered]
            vals = [counts[b] for b in ordered]
            colors = []
            for b in ordered:
                try:
                    colors.append(BEH_COLORS[options.index(b) % len(BEH_COLORS)])
                except ValueError:
                    colors.append(BEH_COLORS[len(colors) % len(BEH_COLORS)])
            short = [b.split(" - ", 1)[-1] if " - " in b else b for b in ordered]
            pct = 100 * sum(vals) / max(total, 1)
            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor(WHITE)
            _donut(ax, short, vals, colors, title,
                   center_text=f"{pct:.0f}%\nannotated")
            handles = [Patch(facecolor=c, edgecolor=WHITE, label=lbl)
                       for c, lbl in zip(colors, short)]
            leg = ax.legend(handles=handles,
                            loc="lower center", bbox_to_anchor=(0.5, -0.14),
                            ncol=3, fontsize=9, framealpha=0.95, edgecolor="#E2E8F0",
                            columnspacing=1.0, handlelength=1.2)
            leg.get_frame().set_linewidth(0.8)
            plt.tight_layout(pad=1.5)
            fig.savefig(out_name, dpi=200, bbox_inches="tight", facecolor=WHITE)
            plt.close(fig)
            saved.append(out_name)

        _axis_donut(mov_timeline, MOVEMENT_OPTIONS, "Movement Composition",
                    f"{base_path}_2_movement.png")
        _axis_donut(soc_timeline, SOCIAL_OPTIONS, "Social Composition",
                    f"{base_path}_2b_social.png")

        # ── 3. Habitat composition — donut chart ──────────────────────
        hab_counts = Counter(lab for lab in hab_timeline if lab)
        if hab_counts:
            ordered = sorted(hab_counts, key=lambda h: hab_counts[h], reverse=True)
            vals    = [hab_counts[h] for h in ordered]
            colors  = [HAB_COLORS.get(h, DEFAULT_CLR) for h in ordered]
            pct     = 100 * sum(vals) / max(total, 1)
            fig, ax = plt.subplots(figsize=(8, 7))
            fig.patch.set_facecolor(WHITE)
            _donut(ax, ordered, vals, colors, "Habitat Composition",
                   center_text=f"{pct:.0f}%\nannotated")
            handles = [Patch(facecolor=c, edgecolor=WHITE, label=lbl)
                       for c, lbl in zip(colors, ordered)]
            leg = ax.legend(handles=handles,
                            loc="lower center", bbox_to_anchor=(0.5, -0.08),
                            ncol=3, fontsize=9, framealpha=0.95, edgecolor="#E2E8F0",
                            columnspacing=1.0, handlelength=1.2)
            leg.get_frame().set_linewidth(0.8)
            plt.tight_layout(pad=1.5)
            out = f"{base_path}_3_habitat.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
            plt.close(fig)
            saved.append(out)

        # ── 3b. Visibility composition — donut chart ──────────────────
        vis_counts = Counter(lab for lab in vis_timeline if lab)
        if vis_counts:
            ordered = [v for v in VISIBILITY_OPTIONS if v in vis_counts]
            ordered += [v for v in vis_counts if v not in ordered]
            vals    = [vis_counts[v] for v in ordered]
            colors  = [VIS_COLORS.get(v, DEFAULT_CLR) for v in ordered]
            pct     = 100 * sum(vals) / max(total, 1)
            fig, ax = plt.subplots(figsize=(8, 7))
            fig.patch.set_facecolor(WHITE)
            _donut(ax, ordered, vals, colors, "Visibility Composition",
                   center_text=f"{pct:.0f}%\nannotated")
            handles = [Patch(facecolor=c, edgecolor=WHITE, label=lbl)
                       for c, lbl in zip(colors, ordered)]
            leg = ax.legend(handles=handles,
                            loc="lower center", bbox_to_anchor=(0.5, -0.08),
                            ncol=3, fontsize=9, framealpha=0.95, edgecolor="#E2E8F0",
                            columnspacing=1.0, handlelength=1.2)
            leg.get_frame().set_linewidth(0.8)
            plt.tight_layout(pad=1.5)
            out = f"{base_path}_3b_visibility.png"
            fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
            plt.close(fig)
            saved.append(out)

        # ── per-habitat plots ──────────────────────────────────────────
        if all_habitats:
            hab_dir = base_path + "_habitat_plots"
            os.makedirs(hab_dir, exist_ok=True)

            for hab in all_habitats:
                hab_slug  = hab.lower().replace(" ", "_")
                hab_color = HAB_COLORS.get(hab, DEFAULT_CLR)

                # ── movement donut for this habitat ──────────────────
                beh_in_hab = Counter(
                    beh_timeline[f]
                    for f in range(total)
                    if hab_timeline[f] == hab and beh_timeline[f]
                )
                if beh_in_hab:
                    ord_b = [b for b in MOVEMENT_OPTIONS if b in beh_in_hab]
                    ord_b += [b for b in beh_in_hab if b not in ord_b]
                    vals_b = [beh_in_hab[b] for b in ord_b]
                    cols_b = []
                    for b in ord_b:
                        try:
                            cols_b.append(BEH_COLORS[MOVEMENT_OPTIONS.index(b)
                                                      % len(BEH_COLORS)])
                        except ValueError:
                            cols_b.append(DEFAULT_CLR)
                    short_b = [b.split(" - ", 1)[-1] if " - " in b else b
                               for b in ord_b]
                    fig, ax = plt.subplots(figsize=(9, 7))
                    fig.patch.set_facecolor(WHITE)
                    _donut(ax, short_b, vals_b, cols_b,
                           f"Movement — {hab}", title_color=hab_color,
                           center_text=f"{sum(vals_b)}\nframes")
                    handles_b = [Patch(facecolor=c, edgecolor=WHITE, label=lbl)
                                 for c, lbl in zip(cols_b, short_b)]
                    leg = ax.legend(handles=handles_b,
                                    loc="lower center", bbox_to_anchor=(0.5, -0.14),
                                    ncol=3, fontsize=9, framealpha=0.95,
                                    edgecolor="#E2E8F0", columnspacing=1.0,
                                    handlelength=1.2)
                    leg.get_frame().set_linewidth(0.8)
                    plt.tight_layout(pad=1.5)
                    out = os.path.join(hab_dir, f"{hab_slug}_behavior.png")
                    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
                    plt.close(fig)
                    saved.append(out)

                # ── species bar for this habitat ─────────────────────
                sp_in_hab  = Counter()
                cat_in_hab = {}
                for feat in store.features:
                    if _feat_habitat(feat) == hab:
                        sp = feat.id_label() or "Unknown"
                        sp_in_hab[sp] += feat.count
                        cat_in_hab[sp] = feat.species_category or "Other"
                if sp_in_hab:
                    sp_list  = sorted(sp_in_hab, key=lambda s: sp_in_hab[s])
                    c_list   = [sp_in_hab[s] for s in sp_list]
                    col_list = [CAT_COLORS.get(cat_in_hab.get(s, "Other"), DEFAULT_CLR)
                                for s in sp_list]
                    n_sp   = len(sp_list)
                    height = max(4.0, n_sp * 0.48 + 2.0)
                    fig, ax = plt.subplots(figsize=(10, height))
                    fig.patch.set_facecolor(WHITE)
                    ax.set_facecolor(BG)
                    _hbar(ax, sp_list, c_list, col_list, cat_in_hab,
                          f"Species — {hab}", title_color=hab_color)
                    _fish_badge(ax, sp_list, sp_in_hab, cat_in_hab)
                    plt.tight_layout(pad=1.8)
                    out = os.path.join(hab_dir, f"{hab_slug}_species.png")
                    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
                    plt.close(fig)
                    saved.append(out)

        # ── 4. Sightings over time with habitat shading ───────────────
        import matplotlib.ticker as mticker
        from matplotlib.transforms import blended_transform_factory

        fps        = max(float(self.fps), 1e-6)
        total_secs = total / fps

        window_s      = 14.0
        window_frames = max(1, min(int(window_s * fps), total))

        new_sightings = np.zeros(total, dtype=float)
        for feat in store.features:
            if 0 <= feat.init_frame < total:
                new_sightings[feat.init_frame] += feat.count

        kernel       = np.ones(window_frames)
        rolling_sum  = np.convolve(new_sightings, kernel, mode="same")
        window_mins  = window_frames / fps / 60.0
        rolling_rate = rolling_sum / max(window_mins, 1e-9)
        times        = np.arange(total) / fps

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor(WHITE)
        ax.set_facecolor(BG)

        segs = FeatureStore._compress_timeline(hab_timeline)
        for seg in segs:
            v = seg.get("value")
            if v is None:
                continue
            t_s = seg["start"] / fps
            t_e = (seg["end"] + 1) / fps
            c   = HAB_COLORS.get(v, DEFAULT_CLR)
            ax.axvspan(t_s, t_e, color=c, alpha=0.15, zorder=0, lw=0)
            if seg["start"] > 0:
                ax.axvline(t_s, color=c, linewidth=1.0,
                           linestyle="--", alpha=0.5, zorder=1)

        ax.plot(times, rolling_rate, color="#4F46E5", linewidth=2.2, zorder=3)
        ax.fill_between(times, rolling_rate, alpha=0.14, color="#4F46E5", zorder=2)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, times[-1] if len(times) > 1 else 1)

        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for seg in segs:
            v = seg.get("value")
            if v is None:
                continue
            t_s = seg["start"] / fps
            t_e = (seg["end"] + 1) / fps
            if (t_e - t_s) < total_secs * 0.04:
                continue
            c = HAB_COLORS.get(v, DEFAULT_CLR)
            ax.text((t_s + t_e) / 2, 0.97, v,
                    ha="center", va="top", transform=trans,
                    fontsize=9, fontweight="600", color=c)

        def _fmt_time(x, _):
            s = int(max(0, x))
            return f"{s // 60}:{s % 60:02d}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(10, integer=False))

        ax.set_title(f"Animal Sightings Over Time  (rolling {window_s:.0f} s window)",
                     fontsize=15, fontweight="bold", pad=14, color="#0F172A")
        ax.set_xlabel("Time (m:ss)", fontsize=11, labelpad=8, color="#334155")
        ax.set_ylabel("Sightings / minute", fontsize=11, labelpad=8, color="#334155")
        ax.tick_params(axis="both", colors="#475569", labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        # Habitat color legend
        hab_seen = sorted({seg["value"] for seg in segs if seg.get("value")})
        if hab_seen:
            from matplotlib.patches import Patch as _Patch
            handles = [_Patch(facecolor=HAB_COLORS.get(h, DEFAULT_CLR),
                              alpha=0.55, edgecolor="none", label=h)
                       for h in hab_seen]
            leg = ax.legend(handles=handles, title="Habitat",
                            fontsize=9, title_fontsize=10,
                            framealpha=0.95, edgecolor="#E2E8F0",
                            loc="upper right")
            leg.get_frame().set_linewidth(0.8)

        plt.tight_layout(pad=1.8)
        out = f"{base_path}_4_sightings_over_time.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=WHITE)
        plt.close(fig)
        saved.append(out)

        # ── 5. Per-species sighting timelines ─────────────────────────
        if species_counts and total > 1:
            sp_dir = base_path + "_species_timelines"
            os.makedirs(sp_dir, exist_ok=True)

            def _timeline_ax(ax, sightings_arr, line_color, title):
                roll = np.convolve(sightings_arr, np.ones(window_frames),
                                   mode="same") / max(window_mins, 1e-9)
                for seg in segs:
                    v = seg.get("value")
                    if v is None:
                        continue
                    t_s = seg["start"] / fps
                    t_e = (seg["end"] + 1) / fps
                    c   = HAB_COLORS.get(v, DEFAULT_CLR)
                    ax.axvspan(t_s, t_e, color=c, alpha=0.15, zorder=0, lw=0)
                    if seg["start"] > 0:
                        ax.axvline(t_s, color=c, linewidth=1.0,
                                   linestyle="--", alpha=0.5, zorder=1)
                ax.plot(times, roll, color=line_color, linewidth=2.2, zorder=3)
                ax.fill_between(times, roll, alpha=0.14,
                                color=line_color, zorder=2)
                ax.set_ylim(bottom=0)
                ax.set_xlim(0, times[-1] if len(times) > 1 else 1)
                trans = blended_transform_factory(ax.transData, ax.transAxes)
                for seg in segs:
                    v = seg.get("value")
                    if v is None:
                        continue
                    t_s = seg["start"] / fps
                    t_e = (seg["end"] + 1) / fps
                    if (t_e - t_s) < total_secs * 0.04:
                        continue
                    ax.text((t_s + t_e) / 2, 0.97, v,
                            ha="center", va="top", transform=trans,
                            fontsize=9, fontweight="600",
                            color=HAB_COLORS.get(v, DEFAULT_CLR))
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_time))
                ax.xaxis.set_major_locator(mticker.MaxNLocator(10, integer=False))
                ax.set_title(title, fontsize=13, fontweight="bold",
                             pad=12, color="#0F172A")
                ax.set_xlabel("Time (m:ss)", fontsize=10, labelpad=6,
                              color="#334155")
                ax.set_ylabel("Sightings / minute", fontsize=10,
                              labelpad=6, color="#334155")
                ax.tick_params(axis="both", colors="#475569", labelsize=9)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.spines["left"].set_color("#CBD5E1")
                ax.spines["bottom"].set_color("#CBD5E1")
                ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.7, zorder=0)
                ax.set_axisbelow(True)
                if hab_seen:
                    from matplotlib.patches import Patch as _P
                    hleg = ax.legend(
                        handles=[_P(facecolor=HAB_COLORS.get(h, DEFAULT_CLR),
                                    alpha=0.55, edgecolor="none", label=h)
                                 for h in hab_seen],
                        title="Habitat", fontsize=8, title_fontsize=9,
                        framealpha=0.95, edgecolor="#E2E8F0",
                        loc="upper right")
                    hleg.get_frame().set_linewidth(0.8)

            for sp_name in sorted(species_counts):
                sp_arr = np.zeros(total, dtype=float)
                for feat in store.features:
                    key = feat.id_label() or "Unknown"
                    if key == sp_name and 0 <= feat.init_frame < total:
                        sp_arr[feat.init_frame] += feat.count
                if not sp_arr.any():
                    continue
                cat        = cat_for_species.get(sp_name, "Other")
                line_color = CAT_COLORS.get(cat, DEFAULT_CLR)
                slug       = re.sub(r"[^\w]+", "_", sp_name).strip("_").lower()
                fig, ax    = plt.subplots(figsize=(13, 4))
                fig.patch.set_facecolor(WHITE)
                ax.set_facecolor(BG)
                _timeline_ax(ax, sp_arr, line_color,
                             f"{sp_name}  (rolling {window_s:.0f} s window)")
                plt.tight_layout(pad=1.5)
                out = os.path.join(sp_dir, f"{slug}_timeline.png")
                fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=WHITE)
                plt.close(fig)
                saved.append(out)

        return saved

    def _default_save_path(self) -> str:
        """Where the Save-JSON dialog should start.

        If the video is on a cloud mount, default to the guaranteed-local
        annotations folder so saves actually persist; otherwise save next to
        the video as before.
        """
        src = self.store.video_path or self.video_path
        name = (os.path.basename(src.rstrip("/")) or "annotations") + ".annotated.json"
        if is_cloud_path(src):
            os.makedirs(LOCAL_ANNOTATIONS_DIR, exist_ok=True)
            return os.path.join(LOCAL_ANNOTATIONS_DIR, name)
        return (src or self.video_path) + ".annotated.json"

    def _save_json(self):
        src = self.store.video_path or self.video_path
        if is_cloud_path(src):
            self.statusBar().showMessage(
                "This video is on a cloud drive — saving to your local "
                "CTAG_Annotator/annotations folder so it persists reliably.")
        p, _ = QFileDialog.getSaveFileName(
            self, "Save annotations", self._default_save_path(), "JSON (*.json)")
        if not p:
            return
        ok, err = self._safe_save(p)
        if ok:
            self.statusBar().showMessage(f"Saved {p}")
            return
        # Write failed (or didn't materialise) — fall back to a local copy.
        fallback = os.path.join(
            LOCAL_ANNOTATIONS_DIR,
            os.path.basename(p) or (self._source_key() + ".annotated.json"))
        ok2, err2 = self._safe_save(fallback)
        if ok2:
            QMessageBox.warning(
                self, "Saved locally instead",
                f"Couldn't save to:\n{p}\n\n({err})\n\n"
                f"Saved a local copy here instead:\n{fallback}\n\n"
                "Tip: keep annotating locally, then upload this JSON to Drive "
                "via Finder when you're done.")
            self.statusBar().showMessage(f"Saved local copy: {fallback}")
        else:
            QMessageBox.critical(
                self, "Save failed",
                f"Could not save to either location:\n{p}\n({err})\n\n"
                f"{fallback}\n({err2})")

    def _export_video(self):
        if self.playing:
            self.toggle_play()
        p, _ = QFileDialog.getSaveFileName(
            self, "Export annotated video",
            self.video_path + ".annotated.mp4", "MP4 (*.mp4)")
        if not p:
            return
        try:
            self._render(p)
            base = os.path.splitext(p)[0]
            saved_plots = self._save_plots(base)
            msg = f"Exported:\n{p}"
            if saved_plots:
                msg += "\n\n" + self._plots_summary(saved_plots)
            QMessageBox.information(self, "Done", msg)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_plots_only(self):
        if self.playing:
            self.toggle_play()
        p, _ = QFileDialog.getSaveFileName(
            self, "Export plots — choose base filename",
            self.video_path + ".plots", "All files (*)")
        if not p:
            return
        base = os.path.splitext(p)[0] if "." in os.path.basename(p) else p
        try:
            saved_plots = self._save_plots(base)
            if not saved_plots:
                QMessageBox.information(self, "Export Plots", "No plots were generated.")
                return
            QMessageBox.information(self, "Done", self._plots_summary(saved_plots))
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_excel(self):
        if self.playing:
            self.toggle_play()
        try:
            import openpyxl
        except ImportError:
            QMessageBox.warning(self, "Missing dependency",
                                "openpyxl is required for Excel export.\n"
                                "Install with: pip install openpyxl")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "Export Excel",
            self.video_path + ".annotated.xlsx", "Excel (*.xlsx)")
        if not p:
            return
        try:
            wb = self._build_workbook(openpyxl)
            wb.save(p)
            QMessageBox.information(self, "Done", f"Excel saved:\n{p}")
        except Exception as e:
            QMessageBox.critical(self, "Excel Export Error", str(e))

    def _build_workbook(self, openpyxl):
        from openpyxl.chart import BarChart, PieChart, Reference
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from collections import Counter

        wb    = openpyxl.Workbook()
        store = self.store
        fps   = max(float(self.fps), 1e-6)
        total = self.total_frames

        hab_tl = self._propagate(store.habitat_per_frame)
        mov_tl = self._propagate(store.movement_per_frame)
        soc_tl = self._propagate(store.social_per_frame)
        vis_tl = self._propagate(store.visibility_per_frame)

        HEADER_FILL = PatternFill("solid", fgColor="0F172A")
        HEADER_FONT = Font(color="FFFFFF", bold=True, size=10)
        ALT_FILL    = PatternFill("solid", fgColor="F1F5F9")

        def _hrow(ws, cols):
            for c, t in enumerate(cols, 1):
                cell = ws.cell(row=1, column=c, value=t)
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
                cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[1].height = 18

        def _autowidth(ws, mn=8, mx=40):
            for col in ws.columns:
                w = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[
                    get_column_letter(col[0].column)].width = min(mx, max(mn, w + 2))

        def _t(frames):
            s = int(frames / fps)
            return f"{s // 60}:{s % 60:02d}"

        # ── Sheet 1: Sightings flat table ─────────────────────────
        ws1 = wb.active
        ws1.title = "Sightings"
        _hrow(ws1, ["Frame", "Time", "Name", "Category",
                    "Family", "Genus", "Species", "Common Name", "ID Level",
                    "Count", "Males", "Females", "Unknown Sex", "Confidence",
                    "Movement", "Social", "Habitat", "Visibility"])
        ws1.freeze_panes = "A2"
        for r, feat in enumerate(
                sorted(store.features, key=lambda f: f.init_frame), 2):
            fi = feat.init_frame
            sx = feat.sexes or ([feat.sex] if feat.sex else [])
            n_m = sum(1 for s in sx if s == "Male")
            n_f = sum(1 for s in sx if s == "Female")
            n_u = sum(1 for s in sx if s == "Unknown")
            for c, v in enumerate([
                fi, _t(fi), feat.name,
                feat.species_category or "",
                feat.family or "",
                feat.genus or "",
                feat.species or "",
                feat.common or "",
                feat.id_level() or "",
                feat.count,
                n_m, n_f, n_u,
                feat.confidence or "",
                mov_tl[fi] or "",
                soc_tl[fi] or "",
                hab_tl[fi] or "",
                vis_tl[fi] or "",
            ], 1):
                cell = ws1.cell(row=r, column=c, value=v)
                if r % 2 == 0:
                    cell.fill = ALT_FILL
        _autowidth(ws1)

        # ── Sheet 2: Species Summary + bar chart ──────────────────
        ws2 = wb.create_sheet("Species Summary")
        _hrow(ws2, ["Species", "Category", "Count"])
        ws2.freeze_panes = "A2"
        sp_counts = Counter()
        sp_cats   = {}
        for feat in store.features:
            sp = feat.id_label() or "Unknown"
            sp_counts[sp] += feat.count
            sp_cats[sp]    = feat.species_category or "Other"
        sp_list = sorted(sp_counts, key=lambda s: sp_counts[s], reverse=True)
        for r, sp in enumerate(sp_list, 2):
            ws2.cell(row=r, column=1, value=sp)
            ws2.cell(row=r, column=2, value=sp_cats[sp])
            ws2.cell(row=r, column=3, value=sp_counts[sp])
        _autowidth(ws2)
        if sp_list:
            n   = len(sp_list)
            ch  = BarChart()
            ch.type   = "bar"
            ch.barDir = "bar"
            ch.title  = "Species Observed"
            ch.x_axis.title = "Count"
            ch.y_axis.title = "Species"
            ch.style  = 10
            ch.add_data(Reference(ws2, min_col=3, min_row=1, max_row=1+n),
                        titles_from_data=True)
            ch.set_categories(Reference(ws2, min_col=1, min_row=2, max_row=1+n))
            ch.width  = 22
            ch.height = max(10, n * 0.55)
            ws2.add_chart(ch, "E2")

        # ── Summary sheets (Movement / Social / Habitat / Visibility) ─
        def _summary_sheet(name, header, timeline, order, chart_title):
            ws = wb.create_sheet(name)
            _hrow(ws, [header, "Frames", "Duration (s)", "% of Annotated"])
            ws.freeze_panes = "A2"
            counts = Counter(v for v in timeline if v)
            tot = sum(counts.values()) or 1
            if order:
                rows = [x for x in order if x in counts]
                rows += [x for x in counts if x not in rows]
            else:
                rows = sorted(counts, key=lambda x: counts[x], reverse=True)
            for r, x in enumerate(rows, 2):
                f = counts[x]
                ws.cell(row=r, column=1, value=x)
                ws.cell(row=r, column=2, value=f)
                ws.cell(row=r, column=3, value=round(f / fps, 2))
                ws.cell(row=r, column=4, value=round(100 * f / tot, 1))
            _autowidth(ws)
            if rows:
                n = len(rows)
                ch = PieChart()
                ch.title = chart_title
                ch.style = 10
                ch.add_data(Reference(ws, min_col=3, min_row=1, max_row=1 + n),
                            titles_from_data=True)
                ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=1 + n))
                ch.width = 18
                ch.height = 14
                ws.add_chart(ch, "F2")

        _summary_sheet("Movement Summary", "Movement", mov_tl,
                       MOVEMENT_OPTIONS, "Movement Distribution")
        _summary_sheet("Social Summary", "Social", soc_tl,
                       SOCIAL_OPTIONS, "Social Distribution")
        _summary_sheet("Habitat Summary", "Habitat", hab_tl,
                       None, "Habitat Distribution")
        _summary_sheet("Visibility Summary", "Visibility", vis_tl,
                       VISIBILITY_OPTIONS, "Visibility Distribution")

        # ── Timeline sheets (segments) per axis ───────────────────
        def _timeline_sheet(name, header, per_frame):
            ws = wb.create_sheet(name)
            _hrow(ws, ["Start Frame", "Start Time", "End Frame",
                       "End Time", "Duration (s)", header])
            ws.freeze_panes = "A2"
            for r, seg in enumerate(
                    (s for s in FeatureStore._compress_timeline(per_frame)
                     if s.get("value")), 2):
                dur = (seg["end"] - seg["start"] + 1) / fps
                for c, v in enumerate([
                    seg["start"], _t(seg["start"]),
                    seg["end"], _t(seg["end"]),
                    round(dur, 2), seg["value"],
                ], 1):
                    ws.cell(row=r, column=c, value=v)
            _autowidth(ws)

        _timeline_sheet("Movement Timeline", "Movement", store.movement_per_frame)
        _timeline_sheet("Social Timeline", "Social", store.social_per_frame)
        _timeline_sheet("Habitat Timeline", "Habitat", store.habitat_per_frame)
        _timeline_sheet("Visibility Timeline", "Visibility", store.visibility_per_frame)

        # ── Notes sheet ───────────────────────────────────────────
        wsn = wb.create_sheet("Notes")
        _hrow(wsn, ["Frame", "Time", "Comment"])
        wsn.freeze_panes = "A2"
        row = 2
        if store.video_note:
            wsn.cell(row=row, column=1, value="—")
            wsn.cell(row=row, column=2, value="whole video")
            wsn.cell(row=row, column=3, value=store.video_note)
            row += 1
        for note in sorted(store.notes, key=lambda n: n["frame"]):
            fi = int(note["frame"])
            wsn.cell(row=row, column=1, value=fi)
            wsn.cell(row=row, column=2, value=_t(fi))
            wsn.cell(row=row, column=3, value=note.get("text", ""))
            row += 1
        _autowidth(wsn)

        # ── Shark Log sheet — each entry spans its whole social segment ─
        wss = wb.create_sheet("Shark Log")
        _hrow(wss, ["Start Frame", "Start Time", "End Frame", "End Time",
                    "Duration (s)", "Social", "Males", "Females", "Unknown",
                    "Total", "Movement", "Habitat", "Visibility"])
        wss.freeze_panes = "A2"
        for r, e in enumerate(sorted(store.shark_log,
                                     key=lambda x: x["frame"]), 2):
            fi = int(e["frame"])
            seg_s, seg_e, seg_lab = store.social_segment_at(fi)
            dur = (seg_e - seg_s + 1) / fps
            total = e.get("males", 0) + e.get("females", 0) + e.get("unknown", 0)
            for c, v in enumerate([
                seg_s, _t(seg_s), seg_e, _t(seg_e), round(dur, 2),
                seg_lab or "",
                e.get("males", 0), e.get("females", 0), e.get("unknown", 0),
                total,
                mov_tl[fi] or "" if fi < len(mov_tl) else "",
                hab_tl[fi] or "" if fi < len(hab_tl) else "",
                vis_tl[fi] or "" if fi < len(vis_tl) else "",
            ], 1):
                wss.cell(row=r, column=c, value=v)
        _autowidth(wss)

        return wb

    @staticmethod
    def _plots_summary(saved_plots: list) -> str:
        main  = [f for f in saved_plots
                 if "_habitat_plots" not in f and "_species_timelines" not in f]
        hab   = [f for f in saved_plots if "_habitat_plots" in f]
        sp    = [f for f in saved_plots if "_species_timelines" in f]
        lines = [f"{len(saved_plots)} plots saved."]
        if main:
            lines.append("Summary plots:\n" + "\n".join(
                os.path.basename(p) for p in main))
        if hab:
            lines.append(f"{len(hab)} habitat plots:\n{os.path.dirname(hab[0])}")
        if sp:
            lines.append(f"{len(sp)} species timelines:\n{os.path.dirname(sp[0])}")
        return "\n\n".join(lines)

    def _render(self, out_path):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(out_path, fourcc, self.fps, (self.video_w, self.video_h))

        if self.is_frame_dir:
            for fidx in range(self.total_frames):
                frame = cv2.imread(self.frame_files[fidx])
                if frame is None:
                    continue
                for _, feat, mask, bbox in self.store.features_at(fidx):
                    bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
                    self._draw_mask(frame, mask, bgr, self._feature_label(feat), bbox)
                self._draw_scene_overlay(frame, fidx)
                w.write(frame)
        else:
            cap = cv2.VideoCapture(self.video_path)
            fidx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                for _, feat, mask, bbox in self.store.features_at(fidx):
                    bgr = FEATURE_COLORS[feat.color_idx % len(FEATURE_COLORS)][::-1]
                    self._draw_mask(frame, mask, bgr, self._feature_label(feat), bbox)
                self._draw_scene_overlay(frame, fidx)
                w.write(frame)
                fidx += 1
            cap.release()

        w.release()

    @staticmethod
    def _reveal_folder(path):
        try:
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
        except ImportError:
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
        os.makedirs(path, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_local_annotations_folder(self):
        self._reveal_folder(LOCAL_ANNOTATIONS_DIR)

    def _open_config_folder(self):
        self._reveal_folder(CONFIG_DIR)

    # ------------------------------------------------------- focus mode
    def _toggle_focus_mode(self, on=None):
        """Hide everything except the video so it can be watched large."""
        self._focus_mode = (not self._focus_mode) if on is None else bool(on)
        panels = [w for w in (getattr(self, "_right_panel", None),
                              getattr(self, "_bars_host", None),
                              getattr(self, "timeline", None)) if w is not None]
        for w in panels:
            w.setVisible(not self._focus_mode)
        if hasattr(self, "focus_btn"):
            self.focus_btn.setChecked(self._focus_mode)
            self.focus_btn.setText("Exit focus" if self._focus_mode else "Focus")
        self.statusBar().showMessage(
            "Focus mode — press F or the button to bring the panels back."
            if self._focus_mode else "")
        # Redraw only after the layout has actually resized the video pane,
        # otherwise the frame is rescaled to the pre-toggle size and the video
        # doesn't grow into the space the hidden panels just freed up.
        QTimer.singleShot(0, lambda: self._display_frame(self.current_frame_idx))

    # ---------------------------------------------------- saved settings
    def _collect_settings(self):
        return {
            "geometry": [self.width(), self.height()],
            "focus_mode": self._focus_mode,
            "labeling": bool(self._labeling),
            "label_axes": dict(self._label_axes),
            "speed": self.speed_combo.currentText() if hasattr(self, "speed_combo") else "1x",
            "category": self.category_combo.currentText() if hasattr(self, "category_combo") else "",
        }

    def _apply_saved_settings(self):
        st = load_settings()
        if not st:
            return
        axes = st.get("label_axes")
        if isinstance(axes, dict):
            for k, v in axes.items():
                if k in self._label_axes:
                    self._label_axes[k] = bool(v)
                    cb = self._axis_checkboxes.get(k)
                    if cb is not None:
                        cb.blockSignals(True); cb.setChecked(bool(v)); cb.blockSignals(False)
        if "labeling" in st and hasattr(self, "label_mode_btn"):
            on = bool(st["labeling"])
            self._labeling = on
            self.label_mode_btn.setChecked(on)
            self.label_mode_btn.setText("Labeling: ON" if on else "Labeling: OFF")
        spd = st.get("speed")
        if spd and hasattr(self, "speed_combo"):
            i = self.speed_combo.findText(spd)
            if i >= 0:
                self.speed_combo.setCurrentIndex(i)
        cat = st.get("category")
        if cat and hasattr(self, "category_combo"):
            i = self.category_combo.findText(cat)
            if i >= 0:
                self.category_combo.setCurrentIndex(i)
        if st.get("focus_mode"):
            QTimer.singleShot(0, lambda: self._toggle_focus_mode(True))

    def _show_help(self):
        QMessageBox.information(
            self, "CTAG Annotator — Shortcuts & help",
            f"Version {APP_VERSION}\n\n"
            "KEYBOARD SHORTCUTS\n"
            "  ←/→  step 1 frame      Ctrl+←/→  jump 10 frames\n"
            "  Space  play/pause      B  arm Draw Bbox     C  focus Comment box\n"
            "  F  focus mode (video only)\n"
            "  I / O  mark clip In/Out    Delete  remove selected animal\n"
            "  ⌘Z / Ctrl+Z  undo      Ctrl +/-  zoom in/out\n"
            "  Ctrl+S  save JSON      Ctrl+E  export clip     F1  this help\n"
            "  (letter keys are ignored while typing in a text box)\n\n"
            "LABELING\n"
            "  • Movement/Social/Habitat/Visibility bars label the current frame\n"
            "    and persist until changed. The Mov/Soc/Hab/Vis checkboxes pick\n"
            "    which axes are written — untick others to re-do just one (e.g.\n"
            "    fix Sand → Gravel without touching behaviors).\n"
            "  • ＋ on any bar adds a label; Labels menu ▸ Manage labels lets you\n"
            "    add/remove, import a CSV, or restore defaults for behaviors,\n"
            "    habitats and species.\n\n"
            "CONSPECIFIC SHARKS (Social bar)\n"
            "  • Other sharks are a social observation, not a fish-ID box: on the\n"
            "    Social bar set the ♂/♀/? counts and press Log. The entry applies\n"
            "    to the WHOLE social interaction containing that frame (e.g. all\n"
            "    of a 'Brief interaction'), and follows it if you extend it.\n"
            "  • Shown as a dark band + ◆ + count on the Social timeline row and\n"
            "    in the Shark log list (double-click jumps; Delete Entry removes).\n"
            "  • Logging again inside the same interaction replaces its entry.\n\n"
            "OTHER ANIMALS / BOUNDING BOXES\n"
            "  • Draw Bbox (or B), then click TWO corners (right-click cancels).\n"
            "  • Each box belongs to the single frame it was drawn on. Within 1 s\n"
            "    of it you'll see it DASHED with how far away it is — click the\n"
            "    dashed box (or its row in the Boxes list) to jump right to it.\n"
            "  • Click a box (video or list) to EDIT it: the panel turns yellow and\n"
            "    any change to ID/count/confidence/name saves instantly. Drag its\n"
            "    corners to resize. Esc / Done editing returns the panel to your\n"
            "    settings for the next box. Delete removes the selected box\n"
            "    (only when you can see it); ⌘Z brings it back.\n"
            "  • Identify at whatever rank you're sure of: type into Find (or\n"
            "    straight into Species) and Family/Genus back-fill themselves;\n"
            "    or fill only Family (or only Genus) and stop there. The panel\n"
            "    shows which level was recorded. Picking a Family narrows the\n"
            "    Genus and Species lists for browsing.\n\n"
            "COMMENTS & CLIPS\n"
            "  • Add a comment at any frame ('ID this fish later'); rose markers\n"
            "    show them on the timeline. Plus a whole-video note field.\n"
            "  • Mark In/Out → Save Clip → Export Clip (MP4, overlays optional).\n\n"
            "SAVING\n"
            "  • Autosaves ~2 s after every change to ~/CTAG_Annotator/annotations\n"
            "    (local, so it's reliable even when the video streams from Google\n"
            "    Drive). You're prompted to resume on reopen.\n"
            "  • Problems? Error and crash reports are in ~/CTAG_Annotator/logs —\n"
            "    send that folder to Mark.")

    # -------------------------------------------------------------- close
    def closeEvent(self, ev):
        # persist work before closing
        try:
            self._autosave()
        except Exception:
            pass
        try:
            save_settings(self._collect_settings())
        except Exception:
            pass
        if hasattr(self, "autosave_timer"):
            self.autosave_timer.stop()
        if hasattr(self, "_dirty_timer"):
            self._dirty_timer.stop()
        if self.cap is not None:
            self.cap.release()
        if self.temp_dir and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self._cleanup_preloads()
        ev.accept()


# --------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------

def _extract_frames(video_path: str, out_dir: str, show_progress: bool = True) -> int:
    """Extract every frame from video_path as a JPEG into out_dir using cv2.

    Files are named 00000.jpg, 00001.jpg … so MainWindow's frame-dir loader
    can sort them by integer stem. Returns the number of frames written.
    With show_progress=False this touches no Qt objects at all, so it is safe
    to call from a background thread (used for next-video preloading).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    dlg = None
    if show_progress:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        modal = Qt.WindowModality.ApplicationModal if _QT6 else Qt.ApplicationModal
        dlg = QProgressDialog("Extracting frames...", None, 0, max(total, 1))
        dlg.setWindowTitle("Loading video")
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(modal)
        dlg.setValue(0)
        dlg.show()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ok_w = cv2.imwrite(
            os.path.join(out_dir, f"{idx:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not ok_w:
            # Almost always a full disk. Fail loudly instead of leaving a video
            # with silently missing frames.
            cap.release()
            if dlg is not None:
                dlg.close()
            free = shutil.disk_usage(out_dir).free / 1e9
            raise RuntimeError(
                f"Could not write frame {idx} to the temporary folder "
                f"({free:.1f} GB free). Free up disk space and try again.")
        idx += 1
        if dlg is not None:
            dlg.setValue(idx)
            QApplication.processEvents()

    cap.release()
    if dlg is not None:
        dlg.close()
    return idx


def prepare_source(path: str, is_frame_dir: bool, silent: bool = False) -> dict:
    """Resolve a user-chosen path into a frame source + metadata.

    For a video file, frames are extracted to a temp dir (returned as temp_dir);
    the caller owns cleanup.  Returns a dict consumed by MainWindow._switch_source
    and main().  Raises RuntimeError on failure.  silent=True suppresses the Qt
    progress dialog so this can run in a background preload thread.
    """
    if is_frame_dir or os.path.isdir(path):
        if not os.path.isdir(path):
            raise RuntimeError(f"Frames directory not found: {path}")
        jpegs = [fn for fn in os.listdir(path)
                 if fn.lower().endswith(('.jpg', '.jpeg'))]
        if not jpegs:
            raise RuntimeError(f"No JPEG files found in: {path}")
        first_file = sorted(jpegs, key=lambda fn: int(os.path.splitext(fn)[0]))[0]
        first = cv2.imread(os.path.join(path, first_file))
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {first_file}")
        h, w = first.shape[:2]
        frame_files = sorted(
            [os.path.join(path, f) for f in jpegs],
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        return {
            "frames_source": path, "is_frame_dir": True, "temp_dir": None,
            "original": path, "frame_files": frame_files,
            "total_frames": len(jpegs), "video_w": w, "video_h": h, "fps": 30.0,
        }

    # video file → extract frames to a temp dir
    if not os.path.isfile(path):
        raise RuntimeError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    temp_dir = tempfile.mkdtemp(prefix="ctag_frames_")
    try:  # owner marker so a later launch can sweep dirs left by a crash
        with open(os.path.join(temp_dir, ".owner_pid"), "w") as fh:
            fh.write(str(os.getpid()))
    except OSError:
        pass
    try:
        n = _extract_frames(path, temp_dir, show_progress=not silent)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Frame extraction failed: {e}")
    if n == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"No frames could be read from: {path}")
    frame_files = sorted(
        [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
         if f.lower().endswith('.jpg')],
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return {
        "frames_source": temp_dir, "is_frame_dir": True, "temp_dir": temp_dir,
        "original": path, "frame_files": frame_files,
        "total_frames": n, "video_w": w, "video_h": h, "fps": fps,
    }


def _sweep_stale_frame_dirs():
    """Remove temp frame folders left behind by a crashed or killed session.

    Each extracted video is ~GBs of JPEGs in the temp dir; normally they are
    deleted on video switch / exit, but a crash leaks them, and a few leaks can
    fill a laptop disk (which then breaks extraction and saving). Only folders
    whose owning process is gone are removed; these are app-generated caches
    of frames, never user data.
    """
    import glob, time
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "ctag_frames_*")):
        try:
            marker = os.path.join(d, ".owner_pid")
            if os.path.exists(marker):
                pid = int(open(marker).read().strip() or 0)
                try:
                    os.kill(pid, 0)
                    continue            # owner still running
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue
            elif time.time() - os.path.getmtime(d) < 24 * 3600:
                continue                # pre-marker build; leave recent ones
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


def _log_path(name):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError:
        pass
    return os.path.join(LOG_DIR, name)


_FAULT_FILE = None


def _enable_fault_log():
    """Record native crashes (segfaults/aborts) that no Python hook can catch.

    Returns the text of a crash recorded by the PREVIOUS run, if any, after
    archiving it, so the app can tell the user where it is.
    """
    global _FAULT_FILE
    import faulthandler, time
    path = _log_path("hard_crash.log")
    previous = None
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            previous = open(path, errors="replace").read()[-4000:]
            stamp = time.strftime("%Y%m%d-%H%M%S")
            os.replace(path, _log_path(f"hard_crash_{stamp}.log"))
    except OSError:
        pass
    try:
        _FAULT_FILE = open(path, "w")
        _FAULT_FILE.write(f"CTAG Annotator {APP_VERSION} pid {os.getpid()}\n")
        _FAULT_FILE.flush()
        faulthandler.enable(_FAULT_FILE, all_threads=True)
    except Exception:
        pass
    return previous


def _clear_fault_log():
    """Clean exit → the header-only fault log is not a crash report."""
    try:
        if _FAULT_FILE is not None:
            import faulthandler
            faulthandler.disable()
            _FAULT_FILE.close()
            os.remove(_FAULT_FILE.name)
    except Exception:
        pass


def _install_crash_guard():
    """Keep a Python error in a Qt slot from killing the whole app.

    PyQt aborts the process when an exception escapes a slot, so a single
    coding mistake in a handler took the app down mid-annotation and threw
    away everything since the last autosave (exactly how the "crash when you
    place a box" bug behaved). Route those through a hook that flushes work
    to disk first, then tells the user what happened and carries on.
    """
    import traceback
    state = {"showing": False}

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:   # a windowed app bundle may have no usable stderr
            if sys.stderr:
                sys.stderr.write(text)
                sys.stderr.flush()
        except Exception:
            pass
        try:   # keep a permanent record for bug reports
            import time
            with open(_log_path("errors.log"), "a") as fh:
                fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"v{APP_VERSION} ===\n{text}")
        except Exception:
            pass
        saved_to = None
        try:  # emergency save before anything else can go wrong
            for w in QApplication.topLevelWidgets():
                if isinstance(w, MainWindow) and w._has_annotations():
                    ok, _err = w._safe_save(w._autosave_path())
                    if ok:
                        saved_to = w._autosave_path()
        except Exception:
            pass
        if state["showing"]:      # never stack dialogs on repeated errors
            return
        try:
            state["showing"] = True
            note = (f"\n\nYour work was saved to:\n{saved_to}"
                    if saved_to else
                    "\n\nNo unsaved annotations were pending.")
            QMessageBox.critical(
                None, "Something went wrong",
                "The annotator hit an internal error, but it has NOT closed "
                "and your annotations are intact." + note +
                "\n\nPlease send Mark the file:\n" + _log_path("errors.log") +
                "\n\n" + text[-1000:])
        except Exception:
            pass
        finally:
            state["showing"] = False

    sys.excepthook = hook


def _enable_hidpi():
    """Best-effort crispness across Qt5/Qt6."""
    try:
        if _QT6:
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        else:
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="CTAG Annotator — Scene label annotation (no torch/SAM2 required)"
    )
    parser.add_argument("video", nargs="?",
                        help="Path to video file OR directory of JPEG frames")
    parser.add_argument("--frames-dir", action="store_true",
                        help="Input is a directory of JPEG frames (not a video file)")
    args = parser.parse_args()

    # Load user-editable species + behavior lists (seeds CSVs on first run)
    load_all_label_config()

    _enable_hidpi()
    app = QApplication(sys.argv)
    _install_crash_guard()
    previous_crash = _enable_fault_log()
    app.aboutToQuit.connect(_clear_fault_log)
    _sweep_stale_frame_dirs()

    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    _apply_light_palette(app)
    app.setStyleSheet(APP_QSS)

    f = QFont()
    f.setPointSize(12)
    app.setFont(f)

    chosen = args.video
    is_frame_dir = args.frames_dir
    playlist = None

    # No path on the command line → ask what to work on rather than dumping
    # the user straight into a bare file picker.
    if not chosen:
        dlg = StartupDialog()
        if not dlg.exec() or not dlg.result_choice:
            sys.exit(0)
        kind, payload = dlg.result_choice
        if kind == "folder":
            _folder, playlist, chosen = payload
            is_frame_dir = False
        elif kind == "frames":
            chosen, is_frame_dir = payload, True
        else:
            chosen, is_frame_dir = payload, False
    if not chosen:
        sys.exit(0)

    try:
        prep = prepare_source(chosen, is_frame_dir)
    except Exception as e:
        # sys.exit(message) is invisible in the app bundle — the window just
        # never appeared, which looked exactly like a crash.
        QMessageBox.critical(None, "Could not open video",
                             f"{os.path.basename(str(chosen))}\n\n{e}")
        sys.exit(1)

    print(f"Opening annotator: {prep['frames_source']}  "
          f"({prep['total_frames']} frames @ {prep['fps']:.2f} fps)")

    win = MainWindow(
        video_path=prep["frames_source"],
        fps=prep["fps"],
        total_frames=prep["total_frames"],
        video_w=prep["video_w"],
        video_h=prep["video_h"],
        is_frame_dir=prep["is_frame_dir"],
        temp_dir=prep["temp_dir"],
        store_video_path=prep["original"],
    )
    # Whole folder stays in the dropdown (so finished clips remain reachable);
    # we just start on the clip chosen above and preload the adjacent one.
    win._playlist = playlist if playlist else [chosen]
    win._refresh_playlist()
    win._start_preload_next()

    # Size to the screen actually in use rather than a fixed 1360x900, which
    # overflowed smaller laptop displays and pushed the control panel off-screen.
    screen = app.primaryScreen()
    avail = screen.availableGeometry() if screen else None
    if avail is not None:
        w = max(900, min(1360, int(avail.width() * 0.92)))
        h = max(600, min(900, int(avail.height() * 0.92)))
        win.resize(w, h)
        frame = win.frameGeometry()
        frame.moveCenter(avail.center())
        win.move(frame.topLeft())
    else:
        win.resize(1200, 800)
    win.show()
    if previous_crash:
        QTimer.singleShot(400, lambda: QMessageBox.warning(
            win, "The annotator closed unexpectedly last time",
            "Your autosaved work is kept and will be offered when you reopen "
            "that video.\n\nPlease send Mark the crash report in:\n" + LOG_DIR))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
