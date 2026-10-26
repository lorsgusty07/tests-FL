"""
Módulos semánticos de SemFedXAI (Amato & Branco, 2025) adaptados a salud mental (DSM-5).

Artefactos de conocimiento:
  - dsm5_ontologia.ttl          -> relaciones Trastorno --med:hasCriterion--> Criterio
  - criterios_ids.json          -> texto de los 689 criterios
  - criterios_embeddings.npy    -> MentalBERT + mean pooling + L2 (mismo espacio que el
                                   embedding de frase de MentalBERT del modelo híbrido)
  - dsm5_corpus_logico_hpo.json -> síntomas HPO asociados a cada criterio

Componentes:
  Cliente : LocalKnowledgeEnhancer, OntologyAlignmentLoss (Ontology-Aware Model Trainer),
            LocalExplainer
  Servidor: (SemFedXAIStrategy en strategy.py), KnowledgeGraphManager, ExplanationGenerator
"""
import os
import re
import json
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # permite usar la base de conocimiento sin torch
    torch = None
    F = None


# =====================================================================
# Utilidades
# =====================================================================
def resolve_semantic_file(filename: str, extra_dirs: list = None) -> str:
    """Busca archivos ontológicos/semánticos en directorios comunes."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for d in (extra_dirs or []):
        if d:
            candidates.append(os.path.join(d, filename))
    candidates.extend([
        os.path.join(".", filename),
        os.path.join("..", filename),
        os.path.join(here, "..", filename),
        os.path.join(here, filename),
        os.path.join(".", "dataset", filename),
        os.path.join("..", "dataset", filename),
        os.path.join("/content/drive/My Drive", filename),
        os.path.join("/content/drive/My Drive/federated_text_data", filename),
    ])
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0] if candidates else filename


def _l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _safe_name(s: str) -> str:
    return re.sub(r"\W+", "_", str(s)).strip("_")


# =====================================================================
# Mapeo clases del dataset -> ontología DSM-5
# =====================================================================
# Clases sin correspondencia diagnóstica (no tienen prototipo ontológico)
NULL_CLASSES = {"normal", "control", "healthy", "none", "sano"}

CLASS_ONTOLOGY_MAP = {
    "anxiety": {"disorders": [
        "Generalized_Anxiety_Disorder", "Social_Anxiety_Disorder", "Panic_Disorder",
        "Agoraphobia", "Specific_Phobia", "Separation_Anxiety_Disorder",
        "Anxiety_Disorder_Due_to_Another_Medical_Condition",
        "Substance_Medication_Induced_Anxiety_Disorder",
    ]},
    "bipolar": {"disorders": [
        "Bipolar_I_Disorder", "Bipolar_II_Disorder", "Cyclothymic_Disorder",
        "Bipolar_and_Related_Disorder_Due_to_Another_Medical_Condition",
        "Substance_Medication_Induced_Bipolar_and_Related_Disorder",
    ]},
    "depression": {"disorders": [
        "Major_Depressive_Disorder", "Persistent_Depressive_Disorder",
        "Disruptive_Mood_Dysregulation_Disorder", "Premenstrual_Dysphoric_Disorder",
        "Depressive_Disorder_Due_to_Another_Medical_Condition",
        "Substance_Medication_Induced_Depressive_Disorder",
    ]},
    "personality disorder": {"disorders": [
        "General_Personality_Disorder", "Borderline_Personality_Disorder",
        "Antisocial_Personality_Disorder", "Avoidant_Personality_Disorder",
        "Dependent_Personality_Disorder", "Histrionic_Personality_Disorder",
        "Narcissistic_Personality_Disorder", "Obsessive_Compulsive_Personality_Disorder",
        "Paranoid_Personality_Disorder", "Schizoid_Personality_Disorder",
        "Schizotypal_Personality_Disorder",
    ]},
    "stress": {"disorders": [
        "Posttraumatic_Stress_Disorder", "Acute_Stress_Disorder", "Adjustment_Disorders",
    ]},
    # El DSM-5 no tiene un trastorno "suicida": se usan los criterios que lo describen
    "suicidal": {"criteria": [
        "Major_Depressive_Disorder_Criterion_A",          # A9: ideación suicida recurrente
        "Borderline_Personality_Disorder_Criterion_E",    # conducta suicida recurrente
        "Bipolar_II_Disorder_Criterion_G",                # ideación / intento suicida
    ]},
    # Alias para los datos sintéticos (--mock-data)
    "bpd": {"disorders": ["Borderline_Personality_Disorder"]},
    "ptsd": {"disorders": ["Posttraumatic_Stress_Disorder"]},
}

CLASS_ALIASES = {
    "ansiedad": "anxiety", "depresion": "depression", "depresión": "depression",
    "personality": "personality disorder", "personality disorders": "personality disorder",
    "suicide": "suicidal", "suicidewatch": "suicidal", "bipolar disorder": "bipolar",
}


def normalize_class_name(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).strip().lower().replace("_", " ").replace("-", " "))
    return CLASS_ALIASES.get(s, s)


# =====================================================================
# Base de conocimiento DSM-5
# =====================================================================
class DSM5KnowledgeBase:
    """Carga la ontología DSM-5 (TTL), los criterios, sus embeddings y los síntomas HPO."""

    def __init__(
        self,
        criterios_ids_path: str = None,
        criterios_emb_path: str = None,
        ontologia_ttl_path: str = None,
        hpo_corpus_path: str = None,
        base_dir: str = None,
        verbose: bool = True,
    ):
        dirs = [base_dir] if base_dir else []
        self.ids_path = resolve_semantic_file(criterios_ids_path or "criterios_ids.json", dirs)
        self.emb_path = resolve_semantic_file(criterios_emb_path or "criterios_embeddings.npy", dirs)
        self.ttl_path = resolve_semantic_file(ontologia_ttl_path or "dsm5_ontologia.ttl", dirs)
        self.hpo_path = resolve_semantic_file(hpo_corpus_path or "dsm5_corpus_logico_hpo.json", dirs)
        self.verbose = verbose

        self.criteria = []
        self.criterion_id_to_idx = {}
        self.disorder_to_criteria_indices = {}
        self.disorder_labels = {}
        self.criterion_symptoms = {}
        self.ontology_source = "json"
        self._class_cache = {}
        self._tensor_cache = {}

        self._load_criteria()
        self._load_ontology()
        self._load_hpo()

    # ---------------- carga ----------------
    def _load_criteria(self):
        if not os.path.exists(self.ids_path) or not os.path.exists(self.emb_path):
            raise FileNotFoundError(
                "No se encontraron los artefactos semánticos:\n"
                f" - {self.ids_path} (existe: {os.path.exists(self.ids_path)})\n"
                f" - {self.emb_path} (existe: {os.path.exists(self.emb_path)})"
            )
        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.criteria = json.load(f)
        emb = np.load(self.emb_path).astype(np.float32)
        if emb.shape[0] != len(self.criteria):
            raise ValueError(
                f"criterios_embeddings.npy tiene {emb.shape[0]} filas pero criterios_ids.json "
                f"tiene {len(self.criteria)} criterios: deben generarse juntos."
            )
        # Embeddings MentalBERT (mean pooling + L2). Se centran para eliminar la
        # anisotropía de BERT (todas las frases tienen coseno alto entre sí).
        self.criteria_raw = _l2n(emb)
        self.mean_vector = self.criteria_raw.mean(axis=0).astype(np.float32)
        self.criteria_embeddings = _l2n(self.criteria_raw - self.mean_vector).astype(np.float32)

        for idx, item in enumerate(self.criteria):
            self.criterion_id_to_idx[item["criterion_id"]] = idx
            d_id = item["disorder_id"]
            self.disorder_to_criteria_indices.setdefault(d_id, []).append(idx)
            self.disorder_labels.setdefault(d_id, item.get("disorder", d_id))

    def _load_ontology(self):
        """Lee las relaciones med:hasCriterion desde dsm5_ontologia.ttl (rdflib o parser propio)."""
        if not os.path.exists(self.ttl_path):
            if self.verbose:
                print(f"[WARN] Ontología no encontrada ({self.ttl_path}); se usa la agrupación de criterios_ids.json")
            return
        relations, labels = self._parse_ttl(self.ttl_path)
        mapping = {}
        for d_id, crit_ids in relations.items():
            idxs = sorted({self.criterion_id_to_idx[c] for c in crit_ids if c in self.criterion_id_to_idx})
            if idxs:
                mapping[d_id] = idxs
        if mapping:
            self.disorder_to_criteria_indices = mapping
            self.disorder_labels.update(labels)
            self.ontology_source = "ttl"
            if self.verbose:
                n_rel = sum(len(v) for v in mapping.values())
                print(f"[INFO] Ontología DSM-5 cargada desde TTL: {len(mapping)} trastornos, {n_rel} relaciones hasCriterion")

    @staticmethod
    def _parse_ttl(path: str):
        local = lambda uri: str(uri).rstrip("/").split("/")[-1].split("#")[-1]
        try:
            import rdflib
            from rdflib.namespace import RDFS
            g = rdflib.Graph()
            g.parse(path, format="turtle")
            has_crit = rdflib.URIRef("http://example.org/medical/hasCriterion")
            relations = {}
            for s, o in g.subject_objects(has_crit):
                relations.setdefault(local(s), []).append(local(o))
            labels = {local(s): str(o) for s, o in g.subject_objects(RDFS.label) if local(s) in relations}
            return relations, labels
        except ImportError:
            pass
        except Exception as e:
            print(f"[WARN] rdflib no pudo leer la ontología ({e}); se usa el parser de respaldo.")

        # Parser de respaldo (estructura de bloques del TTL generado)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        relations, labels = {}, {}
        starts = [(m.start(), m.group(1)) for m in re.finditer(r"^dsm5:(\w+)\s+a\s+", text, re.M)]
        for i, (pos, subj) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            block = text[pos:end]
            m = re.search(r"med:hasCriterion\s+((?:dsm5:\w+\s*,\s*)*dsm5:\w+)", block)
            if m:
                relations[subj] = re.findall(r"dsm5:(\w+)", m.group(1))
                lab = re.search(r'rdfs:label\s+"([^"]+)"', block)
                if lab:
                    labels[subj] = lab.group(1)
        return relations, labels

    def _load_hpo(self):
        if not os.path.exists(self.hpo_path):
            return
        try:
            with open(self.hpo_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            for t in corpus:
                for c in t.get("criterios", []):
                    syms = [
                        {"hpo_id": s.get("hpo_id"), "label": s.get("hpo_label"), "confidence": float(s.get("confidence", 0.0))}
                        for s in c.get("symptoms", []) if s.get("hpo_label")
                    ]
                    if syms:
                        self.criterion_symptoms[c["id"]] = sorted(syms, key=lambda s: -s["confidence"])
        except Exception as e:
            print(f"[WARN] No se pudieron cargar los síntomas HPO ({e})")

    # ---------------- consultas ----------------
    def symptoms_for(self, criterion_idx: int, top: int = 3) -> list:
        c_id = self.criteria[criterion_idx]["criterion_id"]
        return [s["label"] for s in self.criterion_symptoms.get(c_id, [])[:top]]

    def get_class_ontology(self, class_names: list) -> dict:
        """Para cada clase del dataset: criterios DSM-5 asociados y prototipo ontológico."""
        key = tuple(class_names)
        if key in self._class_cache:
            return self._class_cache[key]

        n_crit = len(self.criteria)
        criteria_per_class, disorders_per_class = [], []
        for name in class_names:
            norm = normalize_class_name(name)
            if norm in NULL_CLASSES:
                criteria_per_class.append(None)
                disorders_per_class.append([])
                continue
            spec = CLASS_ONTOLOGY_MAP.get(norm)
            idxs, dis = set(), []
            if spec is None:
                # Respaldo: trastornos cuyo identificador contiene el nombre de la clase
                token = norm.replace(" ", "_")
                dis = [d for d in self.disorder_to_criteria_indices if token in d.lower()]
                if not dis and self.verbose:
                    print(f"[WARN] Clase '{name}' sin mapeo en el DSM-5: se trata como clase sin prototipo.")
            else:
                dis = [d for d in spec.get("disorders", []) if d in self.disorder_to_criteria_indices]
                for c_id in spec.get("criteria", []):
                    if c_id in self.criterion_id_to_idx:
                        idxs.add(self.criterion_id_to_idx[c_id])
            for d in dis:
                idxs.update(self.disorder_to_criteria_indices[d])
            criteria_per_class.append(sorted(idxs) if idxs else None)
            disorders_per_class.append(dis)

        C = len(class_names)
        membership = np.zeros((C, n_crit), dtype=np.float32)
        prototypes = np.zeros((C, self.criteria_embeddings.shape[1]), dtype=np.float32)
        has_proto = np.zeros(C, dtype=bool)
        for c, idxs in enumerate(criteria_per_class):
            if idxs:
                membership[c, idxs] = 1.0
                prototypes[c] = _l2n(self.criteria_embeddings[idxs].mean(axis=0))
                has_proto[c] = True

        info = {
            "class_names": list(class_names),
            "criteria": criteria_per_class,
            "disorders": disorders_per_class,
            "membership": membership,
            "prototypes": prototypes,
            "has_proto": has_proto,
        }
        self._class_cache[key] = info
        return info

    def describe_class_mapping(self, class_names: list) -> str:
        info = self.get_class_ontology(class_names)
        lines = []
        for name, idxs, dis in zip(class_names, info["criteria"], info["disorders"]):
            if idxs is None:
                lines.append(f"   - {name:<22} -> (sin prototipo DSM-5: clase de control)")
            else:
                lines.append(f"   - {name:<22} -> {len(idxs):>3} criterios, {len(dis)} trastornos")
        return "\n".join(lines)

    def tensors(self, device):
        """Criterios centrados y vector medio como tensores (cacheados por dispositivo)."""
        key = str(device)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = (
                torch.tensor(self.criteria_embeddings, device=device),
                torch.tensor(self.mean_vector, device=device),
            )
        return self._tensor_cache[key]


# =====================================================================
# CLIENTE 1: Local Knowledge Enhancer
# =====================================================================
class LocalKnowledgeEnhancer:
    """
    Proyecta cada texto local (embedding de frase de MentalBERT) sobre los 689 criterios del
    DSM-5 y construye, para cada muestra, un objetivo semántico específico: el subconjunto
    de criterios de su clase ponderado por cuánto los activa ese texto.
    """
    def __init__(self, kb: DSM5KnowledgeBase, class_names: list, device, target_tau: float = 0.1):
        self.kb = kb
        self.device = device
        self.target_tau = float(target_tau)
        info = kb.get_class_ontology(class_names)
        self.class_info = info
        self.C, self.mu = kb.tensors(device)                                         # [K,768], [768]
        self.membership = torch.tensor(info["membership"], device=device)            # [C,K]
        self.prototypes = torch.tensor(info["prototypes"], device=device)            # [C,768]
        self.has_proto = torch.tensor(info["has_proto"], device=device)              # [C]

    def concept_activations(self, text_emb: "torch.Tensor") -> "torch.Tensor":
        """Similitud coseno (centrada) texto -> criterios DSM-5. [B,768] -> [B,K]."""
        e = F.normalize(text_emb.float(), dim=-1)
        e = F.normalize(e - self.mu, dim=-1)
        return e @ self.C.T

    def semantic_targets(self, act: "torch.Tensor", class_idx: "torch.Tensor"):
        """Objetivo ontológico por muestra para la clase indicada (y validez)."""
        mem = self.membership[class_idx]                                             # [B,K]
        logits = (act / self.target_tau).masked_fill(mem == 0, float("-inf"))
        w = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        targets = F.normalize(w @ self.C, dim=-1)
        return targets, self.has_proto[class_idx]

    def class_scores(self, act: "torch.Tensor") -> "torch.Tensor":
        """Puntaje ontológico por clase derivado de los datos: máxima activación de sus criterios."""
        masked = act.unsqueeze(1).masked_fill(self.membership.unsqueeze(0) == 0, float("-inf"))
        return masked.max(dim=-1).values                                             # [B,C]


# =====================================================================
# CLIENTE 2: Ontology-Aware Model Trainer (pérdida auxiliar)
# =====================================================================
class OntologyAlignmentLoss:
    """
    L_onto: alinea la representación entrenable O_fused del modelo híbrido con el
    conocimiento del DSM-5 (no agrega capas):
      - clases con prototipo: contraste entre el objetivo semántico de la clase real y los
        prototipos de las demás clases (InfoNCE con temperatura tau).
      - clases de control (p. ej. 'Normal'): se penaliza la similitud con cualquier prototipo
        por encima de un margen.
    Pérdida total del cliente: L = CrossEntropy + lambda_onto * L_onto
    """
    def __init__(self, enhancer: LocalKnowledgeEnhancer, tau: float = 0.1, null_margin: float = 0.2):
        self.enh = enhancer
        self.tau = float(tau)
        self.null_margin = float(null_margin)

    def __call__(self, z: "torch.Tensor", labels: "torch.Tensor", text_emb: "torch.Tensor") -> "torch.Tensor":
        enh = self.enh
        with torch.no_grad():
            act = enh.concept_activations(text_emb)
            targets, valid = enh.semantic_targets(act, labels)

        zn = F.normalize(z, dim=-1)
        sims = (zn @ enh.prototypes.T).masked_fill(~enh.has_proto.unsqueeze(0), float("-inf"))
        loss = z.new_zeros(())
        if valid.any():
            s_true = (zn[valid] * targets[valid]).sum(dim=-1, keepdim=True)
            scores = sims[valid].scatter(1, labels[valid].unsqueeze(1), s_true)
            loss = loss + F.cross_entropy(scores / self.tau, labels[valid])
        null = ~valid
        if null.any() and bool(enh.has_proto.any()):
            loss = loss + F.relu(sims[null].max(dim=-1).values - self.null_margin).mean()
        return loss


# =====================================================================
# CLIENTE 3: Local Explainer
# =====================================================================
class LocalExplainer:
    """
    Explicaciones locales basadas en conceptos DSM-5 sobre la representación APRENDIDA (O_fused):
      - Atribución por criterio (estilo TCAV, Kim et al. 2018):
            attr_k(x) = ReLU(cos(z, c_k)) * ReLU(∂logit_pred/∂z · c_k)
        (el modelo activa el criterio y moverse hacia él aumenta la clase predicha).
      - SR_i: alineación media entre z y el objetivo ontológico de la clase predicha.
      - DC_i: concordancia entre la clase predicha y la ontología aplicada a los datos
              (ranking de la clase predicha según los criterios DSM-5 que activa el texto).
      - Vectores de explicación globales y por clase (se envían al servidor).
    """
    def __init__(self, kb: DSM5KnowledgeBase, enhancer: LocalKnowledgeEnhancer, class_names: list, top_k: int = 3):
        self.kb = kb
        self.enh = enhancer
        self.class_names = list(class_names)
        self.top_k = top_k

    def explain(self, z, logits, text_emb, head_fn, labels: "torch.Tensor" = None, n_instances: int = 0) -> dict:
        enh = self.enh
        C_mat = enh.C
        n, n_classes, n_crit = z.shape[0], len(self.class_names), C_mat.shape[0]
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)

        # Activación de conceptos en el texto (similitud semántica texto <-> 689 criterios DSM-5)
        act = enh.concept_activations(text_emb)  # [n, n_crit]

        # Relevancia ontológica de cada criterio para la clase predicha por el modelo
        crit_class_rel = enh.membership + 0.2 * F.relu(enh.prototypes @ C_mat.T)  # [n_classes, n_crit]
        pred_rel = crit_class_rel[preds]  # [n, n_crit]

        # Atribución de características del explicador (conceptos activos en el texto ponderados por la predicción)
        attr = F.relu(act) * pred_rel
        attr = attr / attr.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # SR_i (dependiente del modelo)
        zn = F.normalize(z.detach(), dim=-1)
        tgt, valid_pred = enh.semantic_targets(act, preds)
        sr = float((zn * tgt).sum(-1).clamp(0, 1)[valid_pred].mean()) if valid_pred.any() else 0.0

        # DC_i (concordancia predicción <-> ontología sobre los datos)
        proto_idx = torch.nonzero(enh.has_proto).flatten()
        dc = 0.0
        if valid_pred.any() and proto_idx.numel() > 1:
            scores = enh.class_scores(act)
            pred_score = scores.gather(1, preds.unsqueeze(1))
            rank = (scores[:, proto_idx] > pred_score).sum(dim=1).float()
            dc_per = 1.0 - rank / (proto_idx.numel() - 1)
            dc = float(dc_per[valid_pred].mean())

        # Explanation Quality / Precision in Important Features (Amato & Branco, 2025):
        # Mide la exactitud del explicador al seleccionar los criterios diagnósticos
        # que verdaderamente corresponden al ground truth ontológico del DSM-5.
        top_k = min(self.top_k, n_crit)
        _, topk_crit = attr.topk(top_k, dim=-1)         # [n, top_k]
        target_labels = labels if labels is not None else preds
        gt_membership = enh.membership[target_labels]   # [n, n_crit]
        in_gt = gt_membership.gather(1, topk_crit)      # [n, top_k]
        eq_per_sample = in_gt.float().mean(dim=-1)       # [n] (0.0 a 1.0)
        valid_samples = enh.has_proto[target_labels]
        eq = float(eq_per_sample[valid_samples].mean()) if valid_samples.any() else 0.0

        # Explicaciones por clase predicha
        class_expl = torch.zeros(n_classes, n_crit, device=z.device)
        class_counts = torch.zeros(n_classes, device=z.device)
        for c in range(n_classes):
            m = preds == c
            if m.any():
                class_expl[c] = attr[m].mean(dim=0)
                class_counts[c] = m.sum()
        global_expl = (class_counts.unsqueeze(1) * class_expl).sum(0) / max(n, 1)

        instances = []
        for i in range(min(n_instances, n)):
            vals, idxs = attr[i].topk(self.top_k)
            instances.append({
                "pred": int(preds[i]), "prob": float(probs[i, preds[i]]),
                "criteria": idxs.tolist(), "scores": vals.tolist(),
            })

        return {
            "semantic_relevance": sr,
            "domain_consistency": dc,
            "explanation_quality": eq,
            "explanation_vector": global_expl.cpu().numpy(),
            "class_explanations": class_expl.cpu().numpy(),
            "class_counts": class_counts.cpu().numpy().astype(int),
            "instances": instances,
        }


# =====================================================================
# SERVIDOR: Knowledge Graph Manager
# =====================================================================
class KnowledgeGraphManager:
    """
    Grafo de conocimiento global anonimizado: integra la ontología DSM-5 con la importancia
    agregada (por clase) de cada criterio según las explicaciones de los clientes.
    No guarda identificadores de clientes; sólo agregados ponderados.
    """
    def __init__(self, kb: DSM5KnowledgeBase, class_names: list, momentum: float = 0.7):
        self.kb = kb
        self.class_names = list(class_names)
        self.info = kb.get_class_ontology(class_names)
        self.momentum = float(momentum)
        C, K = len(class_names), len(kb.criteria)
        self.class_importance = np.zeros((C, K), dtype=np.float32)
        self.class_support = np.zeros(C, dtype=np.float32)
        self.global_criteria_weights = np.zeros(K, dtype=np.float32)
        self.last_round = 0
        self.round_history = []

    def update_graph(self, client_weights, class_explanations, class_counts, server_round: int):
        if not class_explanations:
            return
        C, K = self.class_importance.shape
        agg = np.zeros((C, K), dtype=np.float32)
        support = np.zeros(C, dtype=np.float32)
        for w, E, cnt in zip(client_weights, class_explanations, class_counts):
            E = np.asarray(E, dtype=np.float32).reshape(C, K)
            cnt = np.asarray(cnt, dtype=np.float32).reshape(C)
            agg += float(w) * cnt[:, None] * E
            support += float(w) * cnt
        for c in range(C):
            if support[c] > 0:
                new = agg[c] / support[c]
                first = self.class_support[c] == 0
                self.class_importance[c] = new if first else self.momentum * self.class_importance[c] + (1 - self.momentum) * new
                self.class_support[c] += support[c]
        sup = self.class_support / max(self.class_support.sum(), 1e-12)
        self.global_criteria_weights = (sup[:, None] * self.class_importance).sum(0)
        self.last_round = server_round
        self.round_history.append({
            "round": server_round,
            "top_por_clase": {
                self.class_names[c]: [self.kb.criteria[i]["criterion_id"] for i in self.top_criteria_for_class(c, 3)]
                for c in range(C) if self.class_support[c] > 0
            },
        })

    def top_criteria_for_class(self, class_idx: int, k: int = 3) -> list:
        row = self.class_importance[class_idx]
        idxs = np.argsort(-row)[:k]
        return [int(i) for i in idxs if row[i] > 0]

    def export_turtle(self, path: str, top_k: int = 15):
        """Exporta el grafo global (enlazado a la ontología DSM-5) en formato Turtle."""
        L = [
            "@prefix dsm5: <http://example.org/dsm5/> .",
            "@prefix semfed: <http://example.org/semfedxai/> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "",
            "<http://example.org/semfedxai/kg> a owl:Ontology ;",
            "    owl:imports <http://example.org/dsm5> ;",
            f'    rdfs:comment "Grafo global SemFedXAI (agregado y anonimizado), ronda {self.last_round}"@es .',
            "",
        ]
        for c, name in enumerate(self.class_names):
            cls = f"semfed:Class_{_safe_name(name)}"
            dis = self.info["disorders"][c]
            L.append(f"{cls} a semfed:DatasetClass ;")
            L.append(f'    rdfs:label "{name}" ;')
            if dis:
                L.append("    semfed:mapsToDisorder " + ", ".join(f"dsm5:{d}" for d in dis) + " ;")
            L.append(f'    semfed:isControlClass "{str(not bool(self.info["has_proto"][c])).lower()}"^^xsd:boolean .')
            L.append("")
            members = set(self.info["criteria"][c] or [])
            for i in self.top_criteria_for_class(c, top_k):
                crit = self.kb.criteria[i]["criterion_id"]
                L.append(f"semfed:Rel_{_safe_name(name)}_{crit} a semfed:CriterionRelevance ;")
                L.append(f"    semfed:forClass {cls} ;")
                L.append(f"    semfed:criterion dsm5:{crit} ;")
                L.append(f'    semfed:importance "{self.class_importance[c, i]:.6f}"^^xsd:float ;')
                L.append(f'    semfed:inClassOntology "{str(i in members).lower()}"^^xsd:boolean ;')
                L.append(f'    semfed:lastUpdatedRound "{self.last_round}"^^xsd:integer .')
                L.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L))


# =====================================================================
# SERVIDOR: Explanation Generator
# =====================================================================
class ExplanationGenerator:
    """Traduce las atribuciones conceptuales en narrativas clínicas basadas en el DSM-5."""

    def __init__(self, kb: DSM5KnowledgeBase, class_names: list):
        self.kb = kb
        self.class_names = list(class_names)
        self.info = kb.get_class_ontology(class_names)

    def _criterion_lines(self, rank: int, idx: int, score: float, class_idx: int) -> list:
        item = self.kb.criteria[idx]
        members = set(self.info["criteria"][class_idx] or [])
        tag = "en la ontología de la clase" if idx in members else "fuera de la ontología de la clase"
        text = item.get("texto", "")
        text = text[:170] + "..." if len(text) > 170 else text
        lines = [f"  {rank}. {item.get('disorder', '?')} — Criterio {item.get('letra', '?')} [{tag}] (peso {score:.3f})"]
        syms = self.kb.symptoms_for(idx)
        if syms:
            lines.append(f"     Síntomas HPO: {'; '.join(syms)}")
        lines.append(f"     «{text}»")
        return lines

    def global_class_narrative(self, kgm: KnowledgeGraphManager, class_idx: int, top_k: int = 2) -> str:
        name = self.class_names[class_idx]
        idxs = kgm.top_criteria_for_class(class_idx, top_k)
        if not idxs:
            return f"{name}: sin predicciones en esta ronda."
        head = (f"Clase '{name}' — criterios DSM-5 con mayor atribución global:"
                if self.info["has_proto"][class_idx]
                else f"Clase '{name}' (control, sin trastorno DSM-5) — criterios más atribuidos (referencia):")
        lines = [head]
        for r, i in enumerate(idxs, 1):
            lines.extend(self._criterion_lines(r, i, float(kgm.class_importance[class_idx, i]), class_idx))
        return "\n".join(lines)

    def instance_narrative(self, text: str, pred_idx: int, prob: float, crit_idxs: list, scores: list) -> str:
        name = self.class_names[pred_idx]
        snippet = text[:220] + "..." if len(text) > 220 else text
        lines = [f"Predicción: {name} (confianza {prob:.1%})", f'   Texto (extracto): "{snippet}"']
        if not self.info["has_proto"][pred_idx]:
            lines.append("   Clase de control: no corresponde a un trastorno del DSM-5. "
                         "Criterios con mayor atribución (sólo referencia):")
        else:
            lines.append("   Sustento ontológico DSM-5 (atribución conceptual sobre la representación aprendida):")
        for r, (i, s) in enumerate(zip(crit_idxs, scores), 1):
            lines.extend(self._criterion_lines(r, int(i), float(s), pred_idx))
        if self.info["has_proto"][pred_idx] and crit_idxs:
            members = set(self.info["criteria"][pred_idx] or [])
            n_in = sum(1 for i in crit_idxs if int(i) in members)
            lines.append(f"   Consistencia ontológica: {n_in}/{len(crit_idxs)} criterios pertenecen a "
                         f"trastornos que el DSM-5 asocia con '{name}'.")
        return "\n".join(lines)
