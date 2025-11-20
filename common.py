from __future__ import annotations
import datetime as dt
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, LightningModule


def fixer_graine(graine: int = 42) -> None:
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)


def tag_maintenant() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def xyxy_vers_xywh(boite: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = boite.tolist()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def creer_dossier(chemin: Path) -> None:
    chemin.mkdir(parents=True, exist_ok=True)


def pil_vers_rgb3_canaux(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ["L", "I;16", "I", "F"]:
        gris = image.convert("L")
        return Image.merge("RGB", (gris, gris, gris))
    return image.convert("RGB")



def entrainer_pour_une_epoque(
    modele: torch.nn.Module,
    optimiseur: torch.optim.Optimizer,
    chargeur,
    appareil: torch.device,
    numero_epoque: int,
    echelle: Optional[torch.amp.GradScaler] = None,
    norme_max: float = 10.0,
) -> float:
    modele.train()
    perte_totale = 0.0
    pas_ok = 0

    barre_progression = tqdm(
        chargeur,
        desc=f"Epoch {numero_epoque} [train]",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for pas, (images, cibles) in enumerate(barre_progression, start=1):
        images = [img.to(appareil) for img in images]
        cibles = [{k: v.to(appareil) for k, v in t.items()} for t in cibles]

        optimiseur.zero_grad(set_to_none=True)

        if echelle is not None:
            with torch.amp.autocast(device_type="cuda"):
                dict_pertes = modele(images, cibles)
                perte = sum(dict_pertes.values())
            if not torch.isfinite(perte):
                barre_progression.set_postfix(
                    loss="nan",
                    avg=f"{(perte_totale / max(1, pas_ok)):.4f}",
                )
                continue
            echelle.scale(perte).backward()
            echelle.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), norme_max)
            echelle.step(optimiseur)
            echelle.update()
        else:
            dict_pertes = modele(images, cibles)
            perte = sum(dict_pertes.values())
            if not torch.isfinite(perte):
                barre_progression.set_postfix(
                    loss="nan",
                    avg=f"{(perte_totale / max(1, pas_ok)):.4f}",
                )
                continue
            perte.backward()
            torch.nn.utils.clip_grad_norm_(modele.parameters(), norme_max)
            optimiseur.step()

        pas_ok += 1
        perte_totale += float(perte.item())
        moyenne = perte_totale / max(1, pas_ok)
        barre_progression.set_postfix(
            loss=f"{float(perte.item()):.4f}",
            avg=f"{moyenne:.4f}",
        )

    return perte_totale / max(1, pas_ok)


@torch.no_grad()
def inferer_et_collecter(
    modele: torch.nn.Module,
    chargeur,
    appareil: torch.device,
    seuil_score: float = 0.05,
) -> List[Dict[str, Any]]:
    modele.eval()
    resultats: List[Dict[str, Any]] = []

    for images, cibles in tqdm(
        chargeur,
        desc="Infer [test]",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    ):
        images = [img.to(appareil) for img in images]
        sorties = modele(images)

        for sortie, cible in zip(sorties, cibles):
            id_image = int(cible["image_id"].item())
            boites = sortie["boxes"].detach().cpu().numpy()
            scores = sortie["scores"].detach().cpu().numpy()

            garder = scores >= seuil_score
            boites = boites[garder]
            scores = scores[garder]

            for b, s in zip(boites, scores):
                resultats.append(
                    {
                        "image_id": id_image,
                        "category_id": 1,
                        "bbox": xyxy_vers_xywh(b),
                        "score": float(s),
                    }
                )

    return resultats



class ModuleDetectionLightning(LightningModule):

    def __init__(
        self,
        modele: torch.nn.Module,
        taux_apprentissage: float,
        decay_poids: float = 1e-4,
    ) -> None:
        super().__init__()
        self.modele = modele
        self.save_hyperparameters(ignore=["modele"])

    def forward(self, images, targets=None):
        if targets is None:
            return self.modele(images)
        return self.modele(images, targets)

    def configure_optimizers(self):
        optimiseur = torch.optim.AdamW(
            self.modele.parameters(),
            lr=self.hparams.taux_apprentissage,
            weight_decay=self.hparams.decay_poids,
        )
        return optimiseur

    def training_step(self, batch, batch_idx: int):
        images, cibles = batch
        dict_pertes = self.modele(images, cibles)
        perte = sum(dict_pertes.values())
        self.log(
            "train_loss",
            perte,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return perte

    def validation_step(self, batch, batch_idx: int):
        images, cibles = batch

        etait_training = self.modele.training
        self.modele.train()

        with torch.no_grad():
            dict_pertes = self.modele(images, cibles)
            perte = sum(dict_pertes.values())

        self.modele.train(etait_training)

        self.log(
            "val_loss",
            perte,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return perte


def creer_trainer_arret_precoce(
    nb_epoques_max: int,
    patience: int = 3,
    petit_delta: float = 0.001,
    surveiller: str = "val_loss",
    mode: str = "min",
    norme_clip_grad: float = 0.0,
    utiliser_amp: bool = False,
    dossier_racine_par_defaut: Optional[Path] = None,
) -> Trainer:
    arret_precoce = EarlyStopping(
        monitor=surveiller,
        patience=patience,
        mode=mode,
        min_delta=petit_delta,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if utiliser_amp and accelerator == "gpu":
        precision = "16-mixed"
    else:
        precision = 32

    entraineur = Trainer(
        max_epochs=nb_epoques_max,
        callbacks=[arret_precoce],
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=norme_clip_grad if norme_clip_grad > 0.0 else 0.0,
        default_root_dir=str(dossier_racine_par_defaut) if dossier_racine_par_defaut is not None else None,
        log_every_n_steps=10,
    )
    return entraineur
