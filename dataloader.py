from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from common import pil_vers_rgb3_canaux


def nettoyer_et_couper_boites(
    boites: List[List[float]],
    largeur: int,
    hauteur: int,
) -> List[List[float]]:
    
    boites_nettoyees: List[List[float]] = []
    for x1, y1, x2, y2 in boites:
        x1 = max(0.0, min(float(x1), float(largeur - 1)))
        y1 = max(0.0, min(float(y1), float(hauteur - 1)))
        x2 = max(0.0, min(float(x2), float(largeur - 1)))
        y2 = max(0.0, min(float(y2), float(hauteur - 1)))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue
        boites_nettoyees.append([x1, y1, x2, y2])
    return boites_nettoyees


class DonneesDetectionHRSID(Dataset):

    def __init__(
        self,
        dossier_images: Path,
        chemin_ann: Path,
        augment: bool = False,
        limit: Optional[int] = None,
    ) -> None:
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
    ) -> Tuple[Image.Image, torch.Tensor]:
        largeur, _ = image.size
        image_retournee = image.transpose(Image.FLIP_LEFT_RIGHT)
        b = boites.clone()
        b[:, [0, 2]] = largeur - boites[:, [2, 0]]
        return image_retournee, b

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

        for a in anns:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            boites.append([x1, y1, x2, y2])
            etiquettes.append(1)
            iscrowd.append(int(a.get("iscrowd", 0)))
            surfaces.append(float(a.get("area", bw * bh)))

        boites = nettoyer_et_couper_boites(boites, largeur, hauteur)

        if boites:
            boites_t = torch.tensor(boites, dtype=torch.float32)
            etiquettes_t = torch.tensor(etiquettes, dtype=torch.int64)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            surfaces_t = torch.tensor(surfaces, dtype=torch.float32)
        else:
            boites_t = torch.zeros((0, 4), dtype=torch.float32)
            etiquettes_t = torch.zeros((0,), dtype=torch.int64)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
            surfaces_t = torch.zeros((0,), dtype=torch.float32)

        if self.augment and boites_t.numel() > 0 and random.random() < 0.5:
            image, boites_t = self.retourner_horizontalement(image, boites_t)

        tenseur_image = F.to_tensor(image)

        cible: Dict[str, torch.Tensor] = {
            "boxes": boites_t,
            "labels": etiquettes_t,
            "image_id": torch.tensor([id_image]),
            "iscrowd": iscrowd_t,
            "area": surfaces_t,
        }
        return tenseur_image, cible


def regrouper_pour_batch(batch):
    images, cibles = list(zip(*batch))
    return list(images), list(cibles)
