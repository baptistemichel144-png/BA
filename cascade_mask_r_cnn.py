from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from pycocotools.coco import COCO
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from common import (
    fixer_graine,
    tag_maintenant,
    creer_dossier,
    xyxy_vers_xywh,
)
from results import (
    charger_ids_sousensemble,
    evaluer_coco_sur_sousensembles,
    metriques_a_zero,
    afficher_table_metriques_unique,
    visualiser_image_gt_random,
)

CONFIG = {
    "seed": 42,
    "epochs": 12,
    "ims_per_batch": 2,
    "base_lr": 0.001,
    "num_workers": 4,
    "confidence_threshold": 0.05,
    "print_coco_tables": False,
}


def construire_cfg_cascade_mask_rcnn(
    nom_train: str,
    nom_test: str,
    dossier_sortie: Path,
    chemin_train_ann: Path,
) -> Any:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.DATASETS.TRAIN = (nom_train,)
    cfg.DATASETS.TEST = (nom_test,)
    cfg.DATALOADER.NUM_WORKERS = CONFIG["num_workers"]

    cfg.SOLVER.IMS_PER_BATCH = CONFIG["ims_per_batch"]
    cfg.SOLVER.BASE_LR = CONFIG["base_lr"]
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.STEPS = []

    with open(chemin_train_ann, "r", encoding="utf-8") as f:
        donnees_train = json.load(f)
    nb_images_train = len(donnees_train["images"])
    iters_par_epoque = math.ceil(
        nb_images_train / cfg.SOLVER.IMS_PER_BATCH
    )
    cfg.SOLVER.MAX_ITER = CONFIG["epochs"] * iters_par_epoque

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIG["confidence_threshold"]

    cfg.OUTPUT_DIR = str(dossier_sortie)
    creer_dossier(dossier_sortie)

    return cfg


def faire_inference_et_collecter_coco(
    predictor: DefaultPredictor,
    chemin_test_ann: Path,
    dossier_images: Path,
    seuil_score: float,
) -> List[Dict[str, Any]]:
    
    with open(chemin_test_ann, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    resultats: List[Dict[str, Any]] = []

    for info_image in tqdm(
        images,
        desc="Infer [Cascade Mask R-CNN]",
        leave=True,
        mininterval=0.5,
    ):
        id_image = info_image["id"]
        nom_fichier = info_image["file_name"]
        chemin_image = dossier_images / Path(nom_fichier).name

        if not chemin_image.exists():
            print(f"[warn] Missing image file: {chemin_image}, skipping.")
            continue

        img_pil = Image.open(chemin_image).convert("RGB")
        image = np.array(img_pil)[:, :, ::-1].copy()

        outputs = predictor(image)
        inst = outputs["instances"].to("cpu")

        boites = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        classes = inst.pred_classes.numpy()

        garder = scores >= seuil_score
        boites = boites[garder]
        scores = scores[garder]
        classes = classes[garder]

        for b, s, c in zip(boites, scores, classes):
            if int(c) != 0:
                continue
            resultats.append(
                {
                    "image_id": int(id_image),
                    "category_id": 1,
                    "bbox": xyxy_vers_xywh(b),
                    "score": float(s),
                }
            )

    return resultats


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

    out_root = ici / "outputs"
    creer_dossier(out_root)
    dossier_sortie = out_root / f"cascade_mask_rcnn_{tag_maintenant()}"
    creer_dossier(dossier_sortie)
    print(f"[info] Outputs -> {dossier_sortie}")

    nom_train = "HRSID_train"
    nom_test = "HRSID_test"

    if nom_train not in DatasetCatalog.list():
        register_coco_instances(nom_train, {}, str(chemin_train), str(dossier_images))
    if nom_test not in DatasetCatalog.list():
        register_coco_instances(nom_test, {}, str(chemin_test), str(dossier_images))

    cfg = construire_cfg_cascade_mask_rcnn(nom_train, nom_test, dossier_sortie, chemin_train)

    print("[info] Training Cascade Mask R-CNN with Detectron2...")
    entraineur = DefaultTrainer(cfg)
    entraineur.resume_or_load(resume=False)
    entraineur.train()

    poids_finaux = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    if not poids_finaux.exists():
        print(
            f"[warn] model_final.pth not found in {cfg.OUTPUT_DIR}, "
            "using last checkpoint if any."
        )
    cfg.MODEL.WEIGHTS = str(
        poids_finaux if poids_finaux.exists() else cfg.MODEL.WEIGHTS
    )

    predictor = DefaultPredictor(cfg)

    print("[info] Running inference on test set...")
    detections_coco = faire_inference_et_collecter_coco(
        predictor,
        chemin_test_ann=chemin_test,
        dossier_images=dossier_images,
        seuil_score=CONFIG["confidence_threshold"],
    )

    chemin_json_det = dossier_sortie / "cascade_mask_rcnn_detections_test.json"
    with open(chemin_json_det, "w", encoding="utf-8") as f:
        json.dump(detections_coco, f)
    print(f"[save] Wrote COCO-format detections: {chemin_json_det}")

    coco_verite = COCO(str(chemin_test))

    if len(detections_coco) == 0:
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
        coco_pred = coco_verite.loadRes(str(chemin_json_det))

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

    afficher_table_metriques_unique(
        metriques_all,
        metriques_sub,
        nom_modele="Cascade Mask R-CNN",
    )

    dossier_detections_images = ici / "detections"
    visualiser_image_gt_random(
        chemin_test,
        dossier_images,
        dossier_detections_images,
        prefixe="cascade_mask_rcnn",
        nb_images=10,
        )

    print("\n[done] Cascade Mask R-CNN training + evaluation complete.")


if __name__ == "__main__":
    main()
