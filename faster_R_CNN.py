from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
import torchvision
from pycocotools.coco import COCO
from common import (
    fixer_graine,
    tag_maintenant,
    creer_dossier,
    inferer_et_collecter,
    ModuleDetectionLightning,
    creer_trainer_arret_precoce,
)
from dataloader import DonneesDetectionHRSID, regrouper_pour_batch
from results import (
    charger_ids_sousensemble,
    evaluer_coco_sur_sousensembles,
    metriques_a_zero,
    afficher_table_metriques_unique,
    visualiser_detections,
)


# Experiment configuration for Faster R-CNN on HRSID
CONFIG = {
    "seed": 42,
    "epochs": 12,
    "batch_size_train": 2,
    "batch_size_test": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "confidence_threshold": 0.05,
    "max_images_debug": None,        
    "use_pretrained_backbone": True,
    "amp": True,                     
    "grad_clip_norm": 10.0,
    "print_coco_tables": False,      
    "early_stopping_patience": 3,
    "early_stopping_min_delta": 0.001,
}

# Enable cudnn benchmarking for potential speedup with fixed input sizes
torch.backends.cudnn.benchmark = True



# Model construction: Faster R-CNN with ResNet50-FPN backbone


def construire_faster_rcnn(
    nb_classes: int,
    pretrained_backbone: bool = True,
):
    """
    Build a Faster R-CNN detector with ResNet50-FPN backbone.

    - Attempts to use new torchvision API with explicit weights.
    - Falls back to older API in case of version mismatch.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    try:
        # Newer torchvision style
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models import ResNet50_Weights

        modele = fasterrcnn_resnet50_fpn(
            weights=None,  # no full detector weights, only backbone if set
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
        )
        # Replace classification head to match number of classes (BG + ship)
        in_features = modele.roi_heads.box_predictor.cls_score.in_features
        modele.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            nb_classes,
        )
        return modele
    except Exception:
        # Fallback for older torchvision versions
        modele = fasterrcnn_resnet50_fpn(
            pretrained=False,
            num_classes=nb_classes,
        )
        return modele



# Main pipeline:


def main():
    # 1) Reproducibility
    fixer_graine(CONFIG["seed"])

    # 2) Resolve paths to dataset and outputs
    ici = Path(__file__).resolve().parent

    racine_donnees = ici / "HRSID"
    dossier_annotations = racine_donnees / "annotations"
    dossier_images = racine_donnees / "images"
    dossier_sousensembles = racine_donnees / "inshore_offshore"

    chemin_train = dossier_annotations / "train.json"
    chemin_test = dossier_annotations / "test.json"
    json_inshore = dossier_sousensembles / "inshore.json"
    json_offshore = dossier_sousensembles / "offshore.json"

    # Sanity checks for required files/dirs
    assert chemin_train.exists(), f"Missing: {chemin_train}"
    assert chemin_test.exists(), f"Missing: {chemin_test}"
    assert dossier_images.exists(), f"Missing images dir: {dossier_images}"
    assert json_inshore.exists(), f"Missing: {json_inshore}"
    assert json_offshore.exists(), f"Missing: {json_offshore}"

    # Create a timestamped output directory for this experiment
    dossier_sortie = ici / "outputs" / tag_maintenant()
    creer_dossier(dossier_sortie)
    print(f"[info] Outputs -> {dossier_sortie}")

    # Choose device (GPU if available, else CPU)
    appareil = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[info] Using device: {appareil}")

    # 3) Build training / validation datasets
    donnees_train = DonneesDetectionHRSID(dossier_images, chemin_train, augment=True, limit=limite_train)
    donnees_val = DonneesDetectionHRSID(dossier_images, chemin_test, augment=False, limit=limite_test)

    # 4) Wrap datasets in PyTorch DataLoaders with custom collate function
    chargeur_train = DataLoader(
        donnees_train,
        batch_size=CONFIG["batch_size_train"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=regrouper_pour_batch,
        pin_memory=True,
    )
    chargeur_val = DataLoader(
        donnees_val,
        batch_size=CONFIG["batch_size_test"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=regrouper_pour_batch,
        pin_memory=True,
    )

    # Number of classes = 2 (background and ship)
    nb_classes = 2

    # 5) Build the Faster R-CNN base model
    modele_de_base = construire_faster_rcnn(
        nb_classes=nb_classes,
        pretrained_backbone=CONFIG["use_pretrained_backbone"],
    )

    # 6) Wrap model in a Lightning Module for training
    modele_lightning = ModuleDetectionLightning(
        modele_de_base,
        taux_apprentissage=CONFIG["learning_rate"],
        decay_poids=CONFIG["weight_decay"],
    )

    # 7) Create a Lightning Trainer with EarlyStopping, AMP, grad clipping
    entraineur = creer_trainer_arret_precoce(
        nb_epoques_max=CONFIG["epochs"],
        patience=CONFIG["early_stopping_patience"],
        petit_delta=CONFIG["early_stopping_min_delta"],
        surveiller="val_loss",
        mode="min",
        norme_clip_grad=CONFIG["grad_clip_norm"],
        utiliser_amp=CONFIG["amp"],
        dossier_racine_par_defaut=dossier_sortie,
    )

    # 8) Training loop (Lightning handles epochs / validation / early stopping)
    print("[info] Training Faster R-CNN with PyTorch Lightning + EarlyStopping...")
    entraineur.fit(modele_lightning, train_dataloaders=chargeur_train, val_dataloaders=chargeur_val)

    # 9) Retrieve trained model
    modele = modele_lightning.modele
    modele.to(appareil)

    # 10) Run inference on validation set and collect COCO-style detections
    predictions = inferer_et_collecter(
        modele,
        chargeur_val,
        appareil,
        seuil_score=CONFIG["confidence_threshold"],
    )

    # Save detections to JSON (COCO results format)
    chemin_detections = dossier_sortie / "detections_test.json"
    with open(chemin_detections, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[save] Wrote COCO-format detections: {chemin_detections}")

    # 11) Load COCO ground truth annotations for test set
    coco_verite = COCO(str(chemin_test))

    # 12) Evaluate COCO metrics (overall + inshore/offshore subsets)
    if len(predictions) == 0:
        # no detections at all
        print("[warn] No detections produced; returning zero metrics.")

        ids_tous = coco_verite.getImgIds()
        nom_vers_id_test = {
            Path(img["file_name"]).name: img["id"]
            for img in coco_verite.dataset["images"]
        }

        # Compute image ID lists for inshore/offshore subsets
        ids_inshore = charger_ids_sousensemble(json_inshore, coco_verite, nom_vers_id_test)
        ids_offshore = charger_ids_sousensemble(json_offshore, coco_verite, nom_vers_id_test)

        # Prepare zero-metric dictionaries
        metriques_all: Dict[str, Dict[str, float]] = (
            {"all": metriques_a_zero()} if ids_tous else {}
        )
        metriques_sub: Dict[str, Dict[str, float]] = {
            "inshore": metriques_a_zero() if ids_inshore else metriques_a_zero(),
            "offshore": metriques_a_zero() if ids_offshore else metriques_a_zero(),
        }
    else:
        # Load predictions in COCO API
        coco_pred = coco_verite.loadRes(str(chemin_detections))

        nom_vers_id_test = {
            Path(img["file_name"]).name: img["id"]
            for img in coco_verite.dataset["images"]
        }
        # Build inshore/offshore image ID subsets
        ids_inshore = charger_ids_sousensemble(json_inshore, coco_verite, nom_vers_id_test)
        ids_offshore = charger_ids_sousensemble(json_offshore, coco_verite, nom_vers_id_test)

        # Evaluate on all test images
        ids_tous = coco_verite.getImgIds()
        metriques_all = evaluer_coco_sur_sousensembles(
            coco_verite,
            coco_pred,
            {"all": ids_tous},
            imprimer_tables_coco=CONFIG["print_coco_tables"],
        )
        # Evaluate separately on inshore and offshore subsets
        metriques_sub = evaluer_coco_sur_sousensembles(
            coco_verite,
            coco_pred,
            {"inshore": ids_inshore, "offshore": ids_offshore},
            imprimer_tables_coco=CONFIG["print_coco_tables"],
        )

    # Merge metrics and save to disk
    metriques: Dict[str, Any] = {**metriques_all, **metriques_sub}
    chemin_metriques = dossier_sortie / "metrics.json"
    with open(chemin_metriques, "w", encoding="utf-8") as f:
        json.dump(metriques, f, indent=2)
    print(f"[save] Metrics JSON -> {chemin_metriques}")

    # 13) Visualize a subset of detections on images and save PNGs
    dossier_detections_images = ici / "detections"
    visualiser_detections(
        predictions,
        coco_verite,
        dossier_images,
        dossier_detections_images,
        prefixe="faster_rcnn",
        max_images_a_afficher=10,
        )

    # 14) Print compact summary table (overall / inshore / offshore)
    afficher_table_metriques_unique(
        metriques_all,
        metriques_sub,
        nom_modele="Faster R-CNN",
    )

    print("\n[done] Training + evaluation complete.")


if __name__ == "__main__":
    main()

