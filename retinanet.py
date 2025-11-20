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

torch.backends.cudnn.benchmark = True


def construire_retinanet(nb_classes: int, pretrained_backbone: bool = True):
    from torchvision.models.detection import retinanet_resnet50_fpn

    try:
        from torchvision.models import ResNet50_Weights

        modele = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
            num_classes=nb_classes,
        )
        return modele
    except Exception:
        pass

    try:
        modele = retinanet_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            num_classes=nb_classes,
        )
        return modele
    except TypeError:
        modele = retinanet_resnet50_fpn(pretrained=False, num_classes=nb_classes)
        return modele


def main():
    fixer_graine(CONFIG["seed"])
    ici = Path(__file__).resolve().parent

    racine_donnees = ici / "HRSID"
    dossier_annotations = racine_donnees / "annotations"
    dossier_images = racine_donnees / "images"
    dossier_sousensembles = racine_donnees / "inshore_offshore"

    chemin_train = dossier_annotations / "train.json"
    chemin_test = dossier_annotations / "test.json"
    json_inshore = dossier_sousensembles / "inshore.json"
    json_offshore = dossier_sousensembles / "offshore.json"

    assert chemin_train.exists(), f"Missing: {chemin_train}"
    assert chemin_test.exists(), f"Missing: {chemin_test}"
    assert dossier_images.exists(), f"Missing images dir: {dossier_images}"
    assert json_inshore.exists(), f"Missing: {json_inshore}"
    assert json_offshore.exists(), f"Missing: {json_offshore}"

    dossier_sortie = ici / "outputs" / tag_maintenant()
    creer_dossier(dossier_sortie)
    print(f"[info] Outputs -> {dossier_sortie}")

    appareil = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[info] Using device: {appareil}")

    limite_train = CONFIG["max_images_debug"]
    limite_test = CONFIG["max_images_debug"]

    donnees_train = DonneesDetectionHRSID(dossier_images, chemin_train, augment=True, limit=limite_train)
    donnees_test = DonneesDetectionHRSID(dossier_images, chemin_test, augment=False, limit=limite_test)

    chargeur_train = DataLoader(
        donnees_train,
        batch_size=CONFIG["batch_size_train"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=regrouper_pour_batch,
        pin_memory=True,
    )
    chargeur_val = DataLoader(
        donnees_test,
        batch_size=CONFIG["batch_size_test"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=regrouper_pour_batch,
        pin_memory=True,
    )

    nb_classes = 2

    modele_de_base = construire_retinanet(
        nb_classes=nb_classes,
        pretrained_backbone=CONFIG["use_pretrained_backbone"],
    )

    modele_lightning = ModuleDetectionLightning(
        modele_de_base,
        taux_apprentissage=CONFIG["learning_rate"],
        decay_poids=CONFIG["weight_decay"],
    )

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

    print("[info] Training RetinaNet with PyTorch Lightning + EarlyStopping...")
    entraineur.fit(modele_lightning, train_dataloaders=chargeur_train, val_dataloaders=chargeur_val)

    modele = modele_lightning.modele
    modele.to(appareil)

    predictions = inferer_et_collecter(
        modele,
        chargeur_val,
        appareil,
        seuil_score=CONFIG["confidence_threshold"],
    )

    chemin_detections = dossier_sortie / "detections_test.json"
    with open(chemin_detections, "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"[save] Wrote COCO-format detections: {chemin_detections}")

    coco_verite = COCO(str(chemin_test))

    if len(predictions) == 0:
        print("[warn] No detections produced; returning zero metrics.")

        ids_tous = coco_verite.getImgIds()
        nom_vers_id_test = {
            Path(img["file_name"]).name: img["id"]
            for img in coco_verite.dataset["images"]
        }

        ids_inshore = charger_ids_sousensemble(json_inshore, coco_verite, nom_vers_id_test)
        ids_offshore = charger_ids_sousensemble(json_offshore, coco_verite, nom_vers_id_test)

        metriques_all: Dict[str, Dict[str, float]] = (
            {"all": metriques_a_zero()} if ids_tous else {}
        )
        metriques_sub: Dict[str, Dict[str, float]] = {
            "inshore": metriques_a_zero() if ids_inshore else metriques_a_zero(),
            "offshore": metriques_a_zero() if ids_offshore else metriques_a_zero(),
        }
    else:
        coco_pred = coco_verite.loadRes(str(chemin_detections))

        nom_vers_id_test = {
            Path(img["file_name"]).name: img["id"]
            for img in coco_verite.dataset["images"]
        }
        ids_inshore = charger_ids_sousensemble(json_inshore, coco_verite, nom_vers_id_test)
        ids_offshore = charger_ids_sousensemble(json_offshore, coco_verite, nom_vers_id_test)
        ids_tous = coco_verite.getImgIds()

        metriques_all = evaluer_coco_sur_sousensembles(
            coco_verite,
            coco_pred,
            {"all": ids_tous},
            imprimer_tables_coco=CONFIG["print_coco_tables"],
        )
        metriques_sub = evaluer_coco_sur_sousensembles(
            coco_verite,
            coco_pred,
            {"inshore": ids_inshore, "offshore": ids_offshore},
            imprimer_tables_coco=CONFIG["print_coco_tables"],
        )

    metriques: Dict[str, Any] = {**metriques_all, **metriques_sub}
    chemin_metriques = dossier_sortie / "metrics.json"
    with open(chemin_metriques, "w", encoding="utf-8") as f:
        json.dump(metriques, f, indent=2)
    print(f"[save] Metrics JSON -> {chemin_metriques}")

    dossier_detections_images = ici / "detections"
    visualiser_detections(
        predictions,
        coco_verite,
        dossier_images,
        dossier_detections_images,
        prefixe="retinanet",
        max_images_a_afficher=10,
        )

    afficher_table_metriques_unique(
        metriques_all,
        metriques_sub,
        nom_modele="RetinaNet",
    )

    print("\n[done] Training + evaluation complete.")


if __name__ == "__main__":
    main()
