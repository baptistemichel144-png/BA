from __future__ import annotations
import io
import json
import random
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from common import creer_dossier, pil_vers_rgb3_canaux


def charger_ids_sousensemble(
    json_sousensemble: Path,
    coco_verite: COCO,
    nom_vers_id: Dict[str, int],
) -> List[int]:
    
    with open(json_sousensemble, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids_images: List[int] = []

    def ajouter_par_nom(nom_fichier: str) -> None:
        nom_base = Path(nom_fichier).name
        if nom_base in nom_vers_id:
            ids_images.append(nom_vers_id[nom_base])

    if isinstance(data, dict) and "images" in data and isinstance(
        data["images"], list
    ):
        for item in data["images"]:
            if not isinstance(item, dict):
                continue
            if "id" in item:
                ids_images.append(int(item["id"]))
            elif isinstance(item.get("file_name"), str):
                ajouter_par_nom(item["file_name"])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, int):
                ids_images.append(int(item))
            elif isinstance(item, str):
                ajouter_par_nom(item)
            elif isinstance(item, dict):
                if "id" in item:
                    ids_images.append(int(item["id"]))
                elif isinstance(item.get("file_name"), str):
                    ajouter_par_nom(item["file_name"])

    valides = set(coco_verite.getImgIds())
    ids_images = sorted(list({i for i in ids_images if i in valides}))
    return ids_images



def afficher_entete_coco(nom: str, nb_images: int, type_iou: str = "bbox") -> None:
    display = {
        "all": "OVERALL",
        "overall": "OVERALL",
        "inshore": "INSHORE",
        "offshore": "OFFSHORE",
    }.get(nom.lower(), nom)

    barre = "═" * 80
    print("\n" + barre)
    print(f"COCO {type_iou.upper()} EVALUATION — {display}  |  Images: {nb_images}")
    print(barre)


def avoir_type_iou(coco_eval: COCOeval) -> str:
    return getattr(coco_eval, "iouType", getattr(coco_eval.params, "iouType", "bbox"))


def resumer_silencieusement(coco_eval: COCOeval) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()


def evaluer_coco_sur_sousensembles(
    coco_verite: COCO,
    coco_dets: COCO,
    nom_sousensemble_vers_ids_images: Dict[str, List[int]],
    imprimer_tables_coco: bool = False,
) -> Dict[str, Dict[str, float]]:
    metriques: Dict[str, Dict[str, float]] = {}

    for nom_sousensemble, ids_images in nom_sousensemble_vers_ids_images.items():
        if not ids_images:
            print(f"[warn] No images for subset '{nom_sousensemble}'. Skipping.")
            continue

        coco_eval = COCOeval(coco_verite, coco_dets, iouType="bbox")
        coco_eval.params.imgIds = ids_images

        coco_eval.evaluate()
        coco_eval.accumulate()

        if imprimer_tables_coco:
            afficher_entete_coco(nom_sousensemble, len(ids_images), type_iou=avoir_type_iou(coco_eval))
            coco_eval.summarize()
        else:
            resumer_silencieusement(coco_eval)

        s = coco_eval.stats
        metriques[nom_sousensemble] = {
            "AP": float(s[0]),
            "AP50": float(s[1]),
            "AP75": float(s[2]),
            "APS": float(s[3]),
            "APM": float(s[4]),
            "APL": float(s[5]),
        }

    return metriques


def metriques_a_zero() -> Dict[str, float]:
    return {
        "AP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "APS": 0.0,
        "APM": 0.0,
        "APL": 0.0,
    }



def formater_float(x: float) -> str:
    try:
        if x != x:
            return "nan"
    except Exception:
        pass
    return f"{x:.3f}"


def formater_valeur(nom_ligne: str, nom_col: str, valeur: float) -> str:
    if nom_ligne == "Inshore" and nom_col == "AP_large":
        est_nan = False
        try:
            est_nan = valeur != valeur
        except Exception:
            pass
        if est_nan or (isinstance(valeur, (int, float)) and valeur == -1):
            return "N/A"
    return formater_float(valeur)


def afficher_tableau(entetes: List[str], lignes: List[List[str]]) -> None:
    largeurs = [len(h) for h in entetes]
    for r in lignes:
        for j, cellule in enumerate(r):
            largeurs[j] = max(largeurs[j], len(cellule))

    def fmt_ligne(vals: List[str]) -> str:
        return " | ".join(vals[i].ljust(largeurs[i]) for i in range(len(vals)))

    sep = "-+-".join("-" * w for w in largeurs)
    print(fmt_ligne(entetes))
    print(sep)
    for r in lignes:
        print(fmt_ligne(r))


def afficher_table_metriques_unique(
    metriques_all: Dict[str, Dict[str, float]],
    metriques_sub: Dict[str, Dict[str, float]],
    nom_modele: str = "Model",
) -> None:
    
    map_ligne = {
        "all": "overall",
        "overall": "overall",
        "inshore": "Inshore",
        "offshore": "Offshore",
    }
    ordre = ["overall", "Inshore", "Offshore"]
    fusion = {**metriques_all, **metriques_sub}

    donnees_table: Dict[str, Dict[str, float]] = {r: {} for r in ordre}
    for key, vals in fusion.items():
        nom_ligne = map_ligne.get(key.lower())
        if nom_ligne is None:
            continue
        donnees_table.setdefault(nom_ligne, {})
        donnees_table[nom_ligne]["AP"] = vals.get("AP", float("nan"))
        donnees_table[nom_ligne]["AP_50"] = vals.get("AP50", float("nan"))
        donnees_table[nom_ligne]["AP_75"] = vals.get("AP75", float("nan"))
        donnees_table[nom_ligne]["AP_small"] = vals.get("APS", float("nan"))
        donnees_table[nom_ligne]["AP_medium"] = vals.get("APM", float("nan"))
        donnees_table[nom_ligne]["AP_large"] = vals.get("APL", float("nan"))

    for r in ordre:
        donnees_table.setdefault(
            r,
            {
                "AP": float("nan"),
                "AP_50": float("nan"),
                "AP_75": float("nan"),
                "AP_small": float("nan"),
                "AP_medium": float("nan"),
                "AP_large": float("nan"),
            },
        )

    entetes = ["", "AP", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large"]
    lignes: List[List[str]] = []
    for nom_ligne in ordre:
        vals = donnees_table.get(nom_ligne, {})
        ligne = [
            nom_ligne,
            formater_float(vals.get("AP", float("nan"))),
            formater_float(vals.get("AP_50", float("nan"))),
            formater_float(vals.get("AP_75", float("nan"))),
            formater_float(vals.get("AP_small", float("nan"))),
            formater_float(vals.get("AP_medium", float("nan"))),
            formater_valeur(nom_ligne, "AP_large", vals.get("AP_large", float("nan"))),
        ]
        lignes.append(ligne)

    print("\n" + "=" * 80)
    print(f"HRSID {nom_modele} Results — COCO Metrics (overall / Inshore / Offshore)")
    print("=" * 80)
    afficher_tableau(entetes, lignes)
    print("=" * 80 + "\n")



@torch.no_grad()
def visualiser_detections(
    predictions: List[Dict[str, Any]],
    coco_verite: COCO,
    dossier_images: Path,
    dossier_detections: Path,
    prefixe: str = "model",
    max_images_a_afficher: int = 10,
) -> None:

    if not predictions:
        print("[viz] No detections available for visualization.")
        return

    id_vers_image: Dict[int, Dict[str, Any]] = {
        img["id"]: img for img in coco_verite.dataset.get("images", [])
    }
    image_vers_preds: Dict[int, List[Dict[str, Any]]] = {}
    for det in predictions:
        image_vers_preds.setdefault(int(det["image_id"]), []).append(det)

    ids_candidats = [img_id for img_id, dets in image_vers_preds.items() if dets]
    if not ids_candidats:
        print("[viz] No images with detections to visualize.")
        return

    n = min(max_images_a_afficher, len(ids_candidats))
    if n <= 0:
        return
    ids_selectionnes = (
        random.sample(ids_candidats, n)
        if n < len(ids_candidats)
        else list(ids_candidats)
    )

    creer_dossier(dossier_detections)

    for idx_img, img_id in enumerate(ids_selectionnes, start=1):
        info_image = id_vers_image.get(img_id)
        if info_image is None:
            continue
        nom_fichier = Path(info_image.get("file_name", "")).name
        if not nom_fichier:
            continue
        chemin_image = dossier_images / nom_fichier
        if not chemin_image.exists():
            continue

        image = Image.open(chemin_image)
        image = pil_vers_rgb3_canaux(image)
        largeur, hauteur = image.size
        img_np = np.array(image)

        dets = image_vers_preds.get(img_id, [])
        if not dets:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np)
        nb_boites = 0
        for det in dets:
            x, y, bw, bh = det["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1 = max(0.0, min(float(x1), float(largeur - 1)))
            y1 = max(0.0, min(float(y1), float(hauteur - 1)))
            x2 = max(0.0, min(float(x2), float(largeur - 1)))
            y2 = max(0.0, min(float(y2), float(hauteur - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            nb_boites += 1
        ax.set_title(
            f"{prefixe} detections {idx_img} image_id={img_id} boxes={nb_boites}"
        )
        ax.axis("off")

        chemin_sortie = dossier_detections / f"{prefixe}_detections_{idx_img}.png"
        fig.savefig(chemin_sortie, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] Saved detection visualization to: {chemin_sortie}")


def visualiser_image_gt_random(
    test_ann: Path,
    dossier_images: Path,
    dossier_detections: Path,
    prefixe: str = "cascade_mask_rcnn",
    nb_images: int = 10,
) -> None:
    
    with open(test_ann, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    anns = data.get("annotations", [])
    if not images:
        print("[viz] No images in test annotations.")
        return

    imgid_vers_anns: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        imgid_vers_anns.setdefault(a["image_id"], []).append(a)

    n = min(nb_images, len(images))
    if n <= 0:
        return

    selection = random.sample(images, n) if n < len(images) else list(images)
    creer_dossier(dossier_detections)

    for i, info_image in enumerate(selection, start=1):
        img_id = info_image["id"]
        file_name = info_image["file_name"]
        chemin_image = dossier_images / Path(file_name).name
        if not chemin_image.exists():
            continue

        img_pil = Image.open(chemin_image)
        img_pil = pil_vers_rgb3_canaux(img_pil)
        largeur, hauteur = img_pil.size
        img_np = np.array(img_pil)

        gt_anns = imgid_vers_anns.get(img_id, [])
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_np)
        nb_bateaux = 0
        for a in gt_anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            x1 = max(0.0, min(float(x1), float(largeur - 1)))
            y1 = max(0.0, min(float(y1), float(hauteur - 1)))
            x2 = max(0.0, min(float(x2), float(largeur - 1)))
            y2 = max(0.0, min(float(y2), float(hauteur - 1)))
            if x2 <= x1 or y2 <= y1:
                continue
            rect = Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
            nb_bateaux += 1

        ax.set_title(
            f"{prefixe} GT sample {i} image_id={img_id} ships={nb_bateaux}"
        )
        ax.axis("off")

        chemin_sortie = dossier_detections / f"{prefixe}_gt_sample_{i}.png"
        fig.savefig(chemin_sortie, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved SAR image with boxes to: {chemin_sortie}")
