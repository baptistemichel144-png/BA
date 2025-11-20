from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from common import (
    fixer_graine,
    tag_maintenant,
    creer_dossier,
    pil_vers_rgb3_canaux,
    inferer_et_collecter,
    ModuleDetectionLightning,
    creer_trainer_arret_precoce,
)
from dataloader import regrouper_pour_batch
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


def verifier_et_couper_boite(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    largeur: int,
    hauteur: int,
) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(x1), float(largeur - 1)))
    y1 = max(0.0, min(float(y1), float(hauteur - 1)))
    x2 = max(0.0, min(float(x2), float(largeur - 1)))
    y2 = max(0.0, min(float(y2), float(hauteur - 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    return x1, y1, x2, y2


def masque_rectangle_depuis_boite(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    hauteur: int,
    largeur: int,
) -> np.ndarray:
    masque = np.zeros((hauteur, largeur), dtype=np.uint8)
    x1_i = int(round(x1))
    y1_i = int(round(y1))
    x2_i = int(round(x2))
    y2_i = int(round(y2))
    x1_i = max(0, min(x1_i, largeur - 1))
    y1_i = max(0, min(y1_i, hauteur - 1))
    x2_i = max(0, min(x2_i, largeur - 1))
    y2_i = max(0, min(y2_i, hauteur - 1))
    if x2_i <= x1_i or y2_i <= y1_i:
        return masque
    masque[y1_i:y2_i, x1_i:x2_i] = 1
    return masque


class DonneesMasqueHRSID(Dataset):

    def __init__(
        self,
        dossier_images: Path,
        chemin_ann: Path,
        augment: bool = False,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.dossier_images = Path(dossier_images)
        self.ann = self.charger_json(chemin_ann)
        self.augment = augment

        self.images = self.ann["images"]
        self.annotations = self.ann["annotations"]
        self.categories = self.ann.get("categories", [{"id": 1, "name": "ship"}])

        self.id_vers_image = {img["id"]: img for img in self.images}
        self.nom_vers_id = {
            Path(img["file_name"]).name: img["id"] for img in self.images
        }
        self.image_vers_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations:
            self.image_vers_anns.setdefault(ann["image_id"], []).append(ann)

        self.ids: List[int] = [img["id"] for img in self.images]
        if limit is not None:
            self.ids = self.ids[:limit]

        if self.augment:
            self.ids = [
                i for i in self.ids if len(self.image_vers_anns.get(i, [])) > 0
            ]

    @staticmethod
    def charger_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.ids)

    def retourner_horizontalement(
        self,
        image: Image.Image,
        boites: torch.Tensor,
        masques: torch.Tensor,
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor]:
        largeur, _ = image.size
        image_retournee = image.transpose(Image.FLIP_LEFT_RIGHT)
        b = boites.clone()
        b[:, [0, 2]] = largeur - boites[:, [2, 0]]
        m = torch.flip(masques, dims=[2])
        return image_retournee, b, m

    def __getitem__(self, idx: int):
        id_image = self.ids[idx]
        info_image = self.id_vers_image[id_image]
        nom_fichier = Path(info_image["file_name"]).name
        chemin_image = self.dossier_images / nom_fichier

        image = Image.open(chemin_image)
        image = pil_vers_rgb3_canaux(image)
        largeur, hauteur = image.size

        anns = self.image_vers_anns.get(id_image, [])
        boites: List[List[float]] = []
        etiquettes: List[int] = []
        iscrowd: List[int] = []
        surfaces: List[float] = []
        liste_masques: List[torch.Tensor] = []

        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            clipped = verifier_et_couper_boite(x1, y1, x2, y2, largeur, hauteur)
            if clipped is None:
                continue
            x1_c, y1_c, x2_c, y2_c = clipped
            boites.append([x1_c, y1_c, x2_c, y2_c])
            etiquettes.append(1)
            iscrowd.append(int(a.get("iscrowd", 0)))
            area = float(a.get("area", bw * bh))
            surfaces.append(area)

            masque_np = masque_rectangle_depuis_boite(x1_c, y1_c, x2_c, y2_c, hauteur, largeur)
            liste_masques.append(torch.from_numpy(masque_np))

        if len(boites) > 0:
            boites_t = torch.tensor(boites, dtype=torch.float32)
            etiquettes_t = torch.tensor(etiquettes, dtype=torch.int64)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            surfaces_t = torch.tensor(surfaces, dtype=torch.float32)
            masques_t = (
                torch.stack(liste_masques, dim=0).to(torch.uint8)
                if liste_masques
                else torch.zeros((0, hauteur, largeur), dtype=torch.uint8)
            )
        else:
            boites_t = torch.zeros((0, 4), dtype=torch.float32)
            etiquettes_t = torch.zeros((0,), dtype=torch.int64)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
            surfaces_t = torch.zeros((0,), dtype=torch.float32)
            masques_t = torch.zeros((0, hauteur, largeur), dtype=torch.uint8)

        if self.augment and boites_t.numel() > 0 and random.random() < 0.5:
            image, boites_t, masques_t = self.retourner_horizontalement(image, boites_t, masques_t)

        tenseur_image = F.to_tensor(image)

        cible: Dict[str, Any] = {
            "boxes": boites_t,
            "labels": etiquettes_t,
            "image_id": torch.tensor([id_image]),
            "iscrowd": iscrowd_t,
            "area": surfaces_t,
            "masks": masques_t,
        }
        return tenseur_image, cible


def construire_mask_rcnn(nb_classes: int, pretrained_backbone: bool = True):
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    try:
        from torchvision.models import ResNet50_Weights

        modele = maskrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
            num_classes=nb_classes,
        )
        return modele
    except Exception:
        pass

    try:
        modele = maskrcnn_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            num_classes=nb_classes,
        )
        return modele
    except TypeError:
        modele = maskrcnn_resnet50_fpn(pretrained=False, num_classes=nb_classes)
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

    donnees_train = DonneesMasqueHRSID(dossier_images, chemin_train, augment=True, limit=limite_train)
    donnees_val = DonneesMasqueHRSID(dossier_images, chemin_test, augment=False, limit=limite_test)

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

    nb_classes = 2

    modele_de_base = construire_mask_rcnn(
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

    print("[info] Training Mask R-CNN with PyTorch Lightning + EarlyStopping...")
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
        prefixe="mask_rcnn",
        max_images_a_afficher=10,
        )

    afficher_table_metriques_unique(
        metriques_all,
        metriques_sub,
        nom_modele="Mask R-CNN",
    )

    print("\n[done] Training + evaluation complete.")


if __name__ == "__main__":
    main()
