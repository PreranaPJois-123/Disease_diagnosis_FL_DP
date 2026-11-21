"""Single source of truth for label names/order (docs/03 s3.1) and allowed encoders."""
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
COMP5 = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
COMP5_IDX = [LABELS.index(l) for l in COMP5]
N_LABELS = len(LABELS)
TAB_COLS = ["age_z", "age_missing", "sex_M", "sex_F", "sex_U", "proj_AP", "proj_PA", "proj_O", "prior_studies"]
D_TAB = len(TAB_COLS)
D_IMG = 1024
DEFAULT_POLICY = {"Atelectasis": "ones", "Edema": "ones", "Cardiomegaly": "zeros",
                  "Consolidation": "zeros", "Pleural Effusion": "zeros"}

# TorchXRayVision DenseNet-121 weights. Only these three: none of them saw CheXpert (TEST is CheXpert valid).
ENCODERS = {"nih": "densenet121-res224-nih",
            "mimic_ch": "densenet121-res224-mimic_ch",
            "pc": "densenet121-res224-pc"}
FORBIDDEN_WEIGHTS = ("chex", "all", "rsna", "mimic_nb")   # anything trained on (or mixing) CheXpert is rejected

AGE_BINS = [0, 40, 60, 80, 200]
AGE_LABELS = ["<40", "40-60", "60-80", ">=80"]
