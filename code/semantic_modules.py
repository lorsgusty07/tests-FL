import os
import json
import re
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def resolve_semantic_file(filename: str, extra_dirs: list = None) -> str:
    """Busca archivos ontológicos/semánticos en directorios comunes."""
    candidates = []
    if extra_dirs:
        for d in extra_dirs:
            if d:
                candidates.append(os.path.join(d, filename))
    candidates.extend([
        os.path.join(".", filename),
        os.path.join("..", filename),
        os.path.join(".", "dataset", filename),
        os.path.join("..", "dataset", filename),
        os.path.join("/content/drive/My Drive", filename),
        os.path.join("/content/drive/My Drive/federated_text_data", filename),
        filename
    ])
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return candidates[0]


class DSM5KnowledgeBase:
    """
    Base de Conocimiento Ontológica DSM-5:
    Carga y gestiona dsm5_ontologia.ttl, criterios_ids.json y criterios_embeddings.npy.
    Mapea clases clínicas a trastornos y criterios del DSM-5.
    """
    # Mapeo canónico entre las clases del clasificador y los trastornos del DSM-5
    DEFAULT_CLASS_MAPPING = {
        "Depresion": [
            "Major_Depressive_Disorder",
            "Persistent_Depressive_Disorder",
            "Depressive_Disorder_Due_to_Another_Medical_Condition",
            "Substance_Medication_Induced_Depressive_Disorder"
        ],
        "Ansiedad": [
            "Generalized_Anxiety_Disorder",
            "Social_Anxiety_Disorder",
            "Agoraphobia",
            "Panic_Disorder",
            "Anxiety_Disorder_Due_to_Another_Medical_Condition",
            "Separation_Anxiety_Disorder",
            "Substance_Medication_Induced_Anxiety_Disorder"
        ],
        "BPD": [
            "Borderline_Personality_Disorder"
        ],
        "PTSD": [
            "Posttraumatic_Stress_Disorder",
            "Acute_Stress_Disorder"
        ]
    }

    def __init__(
        self,
        criterios_ids_path: str = None,
        criterios_emb_path: str = None,
        ontologia_ttl_path: str = None,
        base_dir: str = None
    ):
        base_dirs = [base_dir] if base_dir else []
        self.ids_path = resolve_semantic_file(criterios_ids_path or "criterios_ids.json", base_dirs)
        self.emb_path = resolve_semantic_file(criterios_emb_path or "criterios_embeddings.npy", base_dirs)
        self.ttl_path = resolve_semantic_file(ontologia_ttl_path or "dsm5_ontologia.ttl", base_dirs)

        self.criteria = []
        self.criteria_embeddings = None
        self.disorder_to_criteria_indices = {}
        self.criterion_id_to_idx = {}
        self.class_to_criteria_indices = {}

        self.load_data()

    def load_data(self):
        """Carga criterios, embeddings y construye relaciones ontológicas."""
        if not os.path.exists(self.ids_path) or not os.path.exists(self.emb_path):
            raise FileNotFoundError(
                f"No se encontraron los artefactos semánticos necesarios:\n"
                f" - Criterios JSON: {self.ids_path} (Existe: {os.path.exists(self.ids_path)})\n"
                f" - Embeddings NPY: {self.emb_path} (Existe: {os.path.exists(self.emb_path)})"
            )

        with open(self.ids_path, "r", encoding="utf-8") as f:
            self.criteria = json.load(f)

        self.criteria_embeddings = np.load(self.emb_path).astype(np.float32)

        # Normalizar L2 para cálculo eficiente de similitud coseno
        norms = np.linalg.norm(self.criteria_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        self.criteria_embeddings = self.criteria_embeddings / norms

        # Mapear IDs a índices de matriz
        for idx, item in enumerate(self.criteria):
            c_id = item.get("criterion_id")
            d_id = item.get("disorder_id")
            self.criterion_id_to_idx[c_id] = idx

            if d_id not in self.disorder_to_criteria_indices:
                self.disorder_to_criteria_indices[d_id] = []
            self.disorder_to_criteria_indices[d_id].append(idx)

        # Construir mapeo clase -> índices de criterios del DSM-5
        self._build_class_mapping()

    def _build_class_mapping(self):
        """Asocia cada clase clínica de interés con sus criterios correspondientes."""
        for class_name, disorders in self.DEFAULT_CLASS_MAPPING.items():
            indices = []
            for d in disorders:
                if d in self.disorder_to_criteria_indices:
                    indices.extend(self.disorder_to_criteria_indices[d])
            self.class_to_criteria_indices[class_name] = sorted(list(set(indices)))

    def get_criteria_for_class(self, class_name: str) -> list:
        """Devuelve los índices de criterios asociados a una clase diagnóstica."""
        # Búsqueda directa o coincidencia parcial insensible a mayúsculas
        if class_name in self.class_to_criteria_indices:
            return self.class_to_criteria_indices[class_name]
        for k, v in self.class_to_criteria_indices.items():
            if k.lower() in class_name.lower() or class_name.lower() in k.lower():
                return v
        return []

    def compute_similarity(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula la similitud coseno entre representaciones [N, 768] y los 689 criterios [689, 768].
        Retorna matriz [N, 689] con valores en [-1, 1].
        """
        # query_embeddings: [N, 768]
        q_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1e-12
        q_unit = query_embeddings / q_norm
        return np.dot(q_unit, self.criteria_embeddings.T)


class LocalKnowledgeEnhancer:
    """
    Componente 1 de SemFedXAI en el Cliente:
    Local Knowledge Enhancer (LKE).
    Mapea representaciones de características clínicas locales hacia los criterios
    estandarizados del DSM-5 identificando conceptos semánticos activos.
    """
    def __init__(self, knowledge_base: DSM5KnowledgeBase):
        self.kb = knowledge_base

    def enhance_features(self, embeddings: np.ndarray, top_k: int = 5):
        """
        Dadas las representaciones latentes [N, 768], extrae los top_k criterios del DSM-5
        más afines junto con sus puntuaciones de similitud.
        """
        sims = self.kb.compute_similarity(embeddings)  # [N, 689]
        top_k_indices = np.argsort(-sims, axis=1)[:, :top_k]
        top_k_scores = np.take_along_axis(sims, top_k_indices, axis=1)
        return top_k_indices, top_k_scores, sims


class LocalExplainer:
    """
    Componente 3 de SemFedXAI en el Cliente:
    Local Explainer (LE).
    Genera explicaciones clínicas locales a partir de la activación de criterios DSM-5,
    calcula la Relevancia Semántica (SR_i) y la Consistencia con el Dominio (DC_i).
    """
    def __init__(self, knowledge_base: DSM5KnowledgeBase, class_names: list):
        self.kb = knowledge_base
        self.class_names = class_names

    def explain_client_predictions(
        self,
        embeddings: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray = None,
        top_k: int = 5
    ) -> dict:
        """
        Calcula métricas de explicabilidad clínica para el cliente local:
        - SR_i (Semantic Relevance): fuerza promedio de activación de criterios relevantes.
        - DC_i (Domain Consistency): proporción de criterios que pertenecen a la ontología del diagnóstico predicho.
        - explanation_vector: perfil medio de activación [689].
        """
        n_samples = len(predictions)
        if n_samples == 0:
            return {
                "semantic_relevance": 0.0,
                "domain_consistency": 0.0,
                "explanation_vector": [0.0] * len(self.kb.criteria),
                "top_criteria_per_class": {}
            }

        sims = self.kb.compute_similarity(embeddings)  # [N, 689]
        # Restringir a similitudes positivas para atribución
        sims_pos = np.maximum(0.0, sims)

        # Perfil de explicación medio del cliente en R^689
        explanation_vector = np.mean(sims_pos, axis=0)

        # 1. Relevancia Semántica (SR_i): Promedio de las activaciones top_k
        top_scores = np.sort(sims_pos, axis=1)[:, -top_k:]
        sr_i = float(np.mean(top_scores))

        # 2. Consistencia de Dominio (DC_i): % de top-k criterios que pertenecen al trastorno predicho
        valid_criteria_matches = 0
        total_eval_points = n_samples * top_k

        top_indices = np.argsort(-sims_pos, axis=1)[:, :top_k]
        top_per_class = {c: {} for c in self.class_names}

        for idx in range(n_samples):
            pred_idx = int(predictions[idx])
            pred_class = self.class_names[pred_idx] if 0 <= pred_idx < len(self.class_names) else "Desconocido"
            expected_criteria = set(self.kb.get_criteria_for_class(pred_class))

            active_crits = top_indices[idx]
            for c_idx in active_crits:
                if c_idx in expected_criteria:
                    valid_criteria_matches += 1

                # Acumular frecuencia para explicaciones narrativas
                c_id = self.kb.criteria[c_idx]["criterion_id"]
                top_per_class[pred_class][c_id] = top_per_class[pred_class].get(c_id, 0) + 1

        dc_i = float(valid_criteria_matches / total_eval_points) if total_eval_points > 0 else 0.0

        return {
            "semantic_relevance": sr_i,
            "domain_consistency": dc_i,
            "explanation_vector": explanation_vector.tolist(),
            "top_criteria_per_class": top_per_class
        }


class KnowledgeGraphManager:
    """
    Componente 5 de SemFedXAI en el Servidor:
    Knowledge Graph Manager (KGM).
    Mantiene y actualiza el Grafo de Conocimiento global agregando los patrones
    de explicación semántica de los clientes ronda tras ronda.
    """
    def __init__(self, knowledge_base: DSM5KnowledgeBase):
        self.kb = knowledge_base
        self.global_criteria_weights = np.zeros(len(self.kb.criteria), dtype=np.float32)
        self.round_history = []

    def update_graph(self, client_weights: list, client_explanations: list, server_round: int):
        """Actualiza las activaciones globales del Grafo de Conocimiento."""
        if not client_explanations:
            return

        aggregated_vector = np.zeros(len(self.kb.criteria), dtype=np.float32)
        for w, exp in zip(client_weights, client_explanations):
            exp_arr = np.array(exp, dtype=np.float32)
            aggregated_vector += float(w) * exp_arr

        # Acumular y suavizar históricamente
        self.global_criteria_weights = 0.7 * self.global_criteria_weights + 0.3 * aggregated_vector
        self.round_history.append({
            "round": server_round,
            "top_global_indices": np.argsort(-self.global_criteria_weights)[:10].tolist()
        })


class ExplanationGenerator:
    """
    Componente 6 de SemFedXAI en el Servidor:
    Explanation Generator (EG).
    Traduce las métricas y vectores semánticos en narrativas clínicas explicables y
    comprensibles para profesionales de la salud mental basados en el DSM-5.
    """
    def __init__(self, knowledge_base: DSM5KnowledgeBase):
        self.kb = knowledge_base

    def generate_clinical_narrative(
        self,
        explanation_vector: list,
        class_name: str,
        top_k: int = 3
    ) -> str:
        """Genera una explicación clínica estructurada para una clase patológica."""
        exp_arr = np.array(explanation_vector, dtype=np.float32)
        class_criteria = self.kb.get_criteria_for_class(class_name)

        if not class_criteria:
            return f"No se encontraron criterios ontológicos mapeados para la clase {class_name}."

        # Filtrar activaciones para los criterios específicos de esta clase
        scores = [(idx, exp_arr[idx]) for idx in class_criteria]
        scores.sort(key=lambda x: -x[1])

        narrative = [
            f"🩺 [Explicación Clínica DSM-5] Diagnóstico: {class_name.upper()}",
            "Criterios diagnósticos con mayor sustento en las predicciones del modelo:"
        ]

        for rank, (c_idx, score) in enumerate(scores[:top_k], 1):
            item = self.kb.criteria[c_idx]
            disorder = item.get("disorder", "Trastorno")
            letra = item.get("letra", "?")
            texto = item.get("texto", "")
            if len(texto) > 160:
                texto = texto[:160] + "..."
            narrative.append(f"  {rank}. [{disorder} - Criterio {letra}] (Afinidad: {score:.3f}):")
            narrative.append(f"     \"{texto}\"")

        return "\n".join(narrative)
